import os
import json
import logging
import re
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timezone
import time
import redis.asyncio as redis
from dotenv import load_dotenv
import asyncio

load_dotenv()

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.connection_string = os.getenv("REDIS_URL")
        if not self.connection_string:
            raise ValueError("REDIS_URL environment variable is required")
        
        self._client: Optional[redis.Redis] = None

    async def get_pool(self) -> redis.Redis:
        """Get or create Redis client with health check"""
        if self._client is None:
            try:
                self._client = redis.from_url(
                    self.connection_string,
                    decode_responses=True,
                )
                logger.info("Redis client created")
                
                # Verify connection
                await self._verify_connection()
            except Exception as e:
                logger.error(f"Failed to create Redis client: {e}", exc_info=True)
                self._client = None
                raise
        
        # Check if client is still healthy
        try:
            await self._client.ping()
        except Exception as e:
            logger.warning(f"Redis health check failed, recreating client: {e}")
            self._client = None
            return await self.get_pool()
        
        return self._client

    @staticmethod
    def _chat_key(user_id: str, chat_id: str) -> str:
        return f"ai_test2:chat:{user_id}:{chat_id}"

    @staticmethod
    def _user_index_key(user_id: str) -> str:
        return f"ai_test2:user_chats:{user_id}"

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @classmethod
    def _utc_now_iso(cls) -> str:
        return cls._utc_now().isoformat()

    @staticmethod
    def _as_float(value: str, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _parse_chat_hash(self, raw: Dict[str, str]) -> Optional[Dict]:
        if not raw:
            return None

        messages = []
        try:
            if raw.get("messages"):
                messages = json.loads(raw["messages"])
        except (TypeError, json.JSONDecodeError):
            messages = []

        created_at = raw.get("created_at") or self._utc_now_iso()
        updated_at = raw.get("updated_at") or created_at

        return {
            "user_id": raw.get("user_id"),
            "chat_id": raw.get("chat_id"),
            "chat_name": raw.get("chat_name") or None,
            "messages": messages,
            "summary_context": raw.get("summary_context") or None,
            "image_count": int(raw.get("image_count", "0") or 0),
            "created_at": created_at,
            "updated_at": updated_at,
            "updated_at_ts": self._as_float(raw.get("updated_at_ts"), default=time.time()),
        }

    @staticmethod
    def _derive_chat_name_from_prompt(prompt: str, max_words: int = 4) -> str:
        """Create a short chat name from the first user prompt."""
        if not prompt:
            return "New Chat"

        cleaned_prompt = re.sub(r"\s+", " ", prompt.strip())
        cleaned_prompt = re.sub(r"[^\w\s'-]", "", cleaned_prompt)
        words = [w for w in cleaned_prompt.split() if w]
        if not words:
            return "New Chat"

        return " ".join(words[:max_words])[:120]

    async def _get_chat_hash(self, user_id: str, chat_id: str) -> Dict[str, str]:
        client = await self.get_pool()
        return await client.hgetall(self._chat_key(user_id, chat_id))

    async def _save_chat_hash(self, user_id: str, chat_id: str, chat_data: Dict):
        client = await self.get_pool()
        chat_key = self._chat_key(user_id, chat_id)
        user_index_key = self._user_index_key(user_id)
        mapping = {
            "user_id": user_id,
            "chat_id": chat_id,
            "chat_name": chat_data.get("chat_name") or "",
            "messages": json.dumps(chat_data.get("messages", [])),
            "summary_context": chat_data.get("summary_context") or "",
            "image_count": str(chat_data.get("image_count", 0)),
            "created_at": chat_data.get("created_at") or self._utc_now_iso(),
            "updated_at": chat_data.get("updated_at") or self._utc_now_iso(),
            "updated_at_ts": str(chat_data.get("updated_at_ts", time.time())),
        }
        async with client.pipeline(transaction=True) as pipe:
            await pipe.hset(chat_key, mapping=mapping)
            await pipe.zadd(user_index_key, {chat_id: float(mapping["updated_at_ts"])})
            await pipe.execute()

    async def _verify_connection(self):
        """Verify Redis connectivity"""
        try:
            client = await self.get_pool()
            await asyncio.wait_for(client.ping(), timeout=5)
            logger.info("Redis connection verified")
        except Exception as e:
            logger.error(f"Redis connection verification failed: {e}")
            raise

    async def init_database(self):
        """Initialize Redis connectivity with retry handling"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                await self._verify_connection()
                logger.info("Redis storage initialized successfully")
                return
            except Exception as e:
                logger.warning(f"Redis connection error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

    async def get_and_increment_image_count(self, user_id: str, chat_id: str) -> int:
        """Get current image count and increment it for next image"""
        if not user_id or not chat_id:
            raise ValueError("user_id and chat_id are required")
        
        try:
            client = await self.get_pool()
            chat_key = self._chat_key(user_id, chat_id)
            user_index_key = self._user_index_key(user_id)
            now_iso = self._utc_now_iso()
            now_ts = time.time()

            async with client.pipeline(transaction=True) as pipe:
                await pipe.hsetnx(chat_key, "user_id", user_id)
                await pipe.hsetnx(chat_key, "chat_id", chat_id)
                await pipe.hsetnx(chat_key, "messages", "[]")
                await pipe.hsetnx(chat_key, "summary_context", "")
                await pipe.hsetnx(chat_key, "created_at", now_iso)
                await pipe.hincrby(chat_key, "image_count", 1)
                await pipe.hset(chat_key, mapping={"updated_at": now_iso, "updated_at_ts": str(now_ts)})
                await pipe.zadd(user_index_key, {chat_id: now_ts})
                results = await pipe.execute()

            current_count = int(results[5]) if len(results) > 5 else 1
            logger.info(f"Image count for user_id={user_id}, chat_id={chat_id}: {current_count}")
            return current_count
        except Exception as e:
            logger.error(f"Error incrementing image count: {e}", exc_info=True)
            raise

    async def get_chat(self, user_id: str, chat_id: str) -> Optional[Dict]:
        """Get chat data for a specific user and chat"""
        if not user_id or not chat_id:
            raise ValueError("user_id and chat_id are required")
        
        try:
            raw = await self._get_chat_hash(user_id, chat_id)
            chat_data = self._parse_chat_hash(raw)
            if not chat_data:
                return None

            # Backfill chat_name for older chats that were saved before this field existed.
            if not chat_data.get("chat_name"):
                first_user_message = next(
                    (m.get("content", "") for m in chat_data.get("messages", []) if m.get("role") == "user"),
                    ""
                )
                derived_name = self._derive_chat_name_from_prompt(first_user_message)
                now_iso = self._utc_now_iso()
                now_ts = time.time()
                client = await self.get_pool()
                await client.hset(self._chat_key(user_id, chat_id), mapping={
                    "chat_name": derived_name,
                    "updated_at": now_iso,
                    "updated_at_ts": str(now_ts),
                })
                chat_data["chat_name"] = derived_name

            return chat_data
        except Exception as e:
            logger.error(f"Error fetching chat: {e}", exc_info=True)
            raise

    async def get_last_n_messages(self, user_id: str, chat_id: str, n: int = 30) -> Tuple[List[Dict], Optional[str]]:
        """Get last N messages and summary context"""
        if not user_id or not chat_id:
            raise ValueError("user_id and chat_id are required")
        
        if n <= 0:
            raise ValueError("n must be positive")
        
        try:
            chat_data = await self.get_chat(user_id, chat_id)
            if not chat_data:
                return [], None
            
            all_messages = chat_data.get("messages", [])
            last_n = all_messages[-n:] if len(all_messages) > n else all_messages
            summary_context = chat_data.get("summary_context")
            
            return last_n, summary_context
        except Exception as e:
            logger.error(f"Error fetching last N messages: {e}", exc_info=True)
            raise

    async def save_message(self, user_id: str, chat_id: str, role: str, content: str, image_url: str = None):
        """Append a new message to chat_history with optional image URL"""
        if not user_id or not chat_id or not role or not content:
            raise ValueError("user_id, chat_id, role, and content are required")
        
        if role not in ["user", "assistant"]:
            raise ValueError("role must be 'user' or 'assistant'")
        
        try:
            message = {
                "role": role,
                "content": content,
                "timestamp": self._utc_now_iso()
            }
            
            if image_url:
                message["image_url"] = image_url

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    client = await self.get_pool()
                    chat_key = self._chat_key(user_id, chat_id)
                    user_index_key = self._user_index_key(user_id)
                    now_iso = self._utc_now_iso()
                    now_ts = time.time()

                    async with client.pipeline(transaction=True) as pipe:
                        await pipe.watch(chat_key)
                        raw_chat = await pipe.hgetall(chat_key)
                        existing = self._parse_chat_hash(raw_chat) if raw_chat else None

                        messages = existing.get("messages", []) if existing else []
                        messages.append(message)

                        existing_chat_name = (existing or {}).get("chat_name")
                        chat_name = existing_chat_name
                        if not chat_name and role == "user":
                            chat_name = self._derive_chat_name_from_prompt(content)

                        updated_chat = {
                            "chat_name": chat_name or "",
                            "messages": messages,
                            "summary_context": (existing or {}).get("summary_context") or "",
                            "image_count": (existing or {}).get("image_count", 0),
                            "created_at": (existing or {}).get("created_at") or now_iso,
                            "updated_at": now_iso,
                            "updated_at_ts": now_ts,
                        }

                        pipe.multi()
                        await pipe.hset(chat_key, mapping={
                            "user_id": user_id,
                            "chat_id": chat_id,
                            "chat_name": updated_chat["chat_name"],
                            "messages": json.dumps(updated_chat["messages"]),
                            "summary_context": updated_chat["summary_context"],
                            "image_count": str(updated_chat["image_count"]),
                            "created_at": updated_chat["created_at"],
                            "updated_at": updated_chat["updated_at"],
                            "updated_at_ts": str(updated_chat["updated_at_ts"]),
                        })
                        await pipe.zadd(user_index_key, {chat_id: now_ts})
                        await pipe.execute()
                        break
                except redis.WatchError:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(0.05)

            logger.info(f"Message saved for user_id={user_id}, chat_id={chat_id}, role={role}, has_image={image_url is not None}")
        except Exception as e:
            logger.error(f"Error saving message: {e}", exc_info=True)
            raise

    async def save_summary_context(self, user_id: str, chat_id: str, summary_text: str):
        """Save summarized context when chat ends or context gets too large"""
        if not user_id or not chat_id or not summary_text:
            raise ValueError("user_id, chat_id, and summary_text are required")
        
        try:
            client = await self.get_pool()
            chat_key = self._chat_key(user_id, chat_id)
            if not await client.exists(chat_key):
                logger.info(f"Summary context skipped for missing chat: user_id={user_id}, chat_id={chat_id}")
                return

            now_iso = self._utc_now_iso()
            now_ts = time.time()
            async with client.pipeline(transaction=True) as pipe:
                await pipe.hset(chat_key, mapping={
                    "summary_context": summary_text,
                    "updated_at": now_iso,
                    "updated_at_ts": str(now_ts),
                })
                await pipe.zadd(self._user_index_key(user_id), {chat_id: now_ts})
                await pipe.execute()

            logger.info(f"Summary context saved for user_id={user_id}, chat_id={chat_id}")
        except Exception as e:
            logger.error(f"Error saving summary: {e}", exc_info=True)
            raise

    async def get_user_chats(self, user_id: str) -> List[Dict]:
        """Get all chats for a user"""
        if not user_id:
            raise ValueError("user_id is required")
        
        try:
            client = await self.get_pool()
            chat_ids = await client.zrevrange(self._user_index_key(user_id), 0, -1)
            chats: List[Dict] = []

            for chat_id in chat_ids:
                chat_data = await self.get_chat(user_id, chat_id)
                if not chat_data:
                    continue
                chats.append({
                    "chat_id": chat_id,
                    "chat_name": chat_data.get("chat_name") or "New Chat",
                    "created_at": chat_data.get("created_at"),
                    "updated_at": chat_data.get("updated_at"),
                    "message_count": len(chat_data.get("messages", [])),
                })

            return chats
        except Exception as e:
            logger.error(f"Error fetching user chats: {e}", exc_info=True)
            raise

    async def delete_chat(self, user_id: str, chat_id: str) -> bool:
        """Delete a specific chat"""
        if not user_id or not chat_id:
            raise ValueError("user_id and chat_id are required")
        
        try:
            client = await self.get_pool()
            async with client.pipeline(transaction=True) as pipe:
                await pipe.delete(self._chat_key(user_id, chat_id))
                await pipe.zrem(self._user_index_key(user_id), chat_id)
                result = await pipe.execute()

            rows_deleted = int(result[0]) if result else 0
            logger.info(f"Chat deleted: user_id={user_id}, chat_id={chat_id}, rows_deleted={rows_deleted}")
            return rows_deleted > 0
        except Exception as e:
            logger.error(f"Error deleting chat: {e}", exc_info=True)
            raise

    async def close(self):
        """Close Redis client gracefully"""
        if self._client:
            try:
                await self._client.aclose()
                self._client = None
                logger.info("Redis client closed successfully")
            except Exception as e:
                logger.error(f"Error closing Redis client: {e}", exc_info=True)

# Singleton instance
db = Database()
