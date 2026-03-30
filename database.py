import os
import json
import logging
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from psycopg import OperationalError, InterfaceError
from dotenv import load_dotenv
import asyncio

load_dotenv()

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.connection_string = os.getenv("POSTGRES_URL")
        if not self.connection_string:
            raise ValueError("POSTGRES_URL environment variable is required")
        
        self._pool: Optional[AsyncConnectionPool] = None
        self._min_size = int(os.getenv("DB_POOL_MIN_SIZE", 2))
        self._max_size = int(os.getenv("DB_POOL_MAX_SIZE", 10))
        self._pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", 30))
        self._query_timeout = int(os.getenv("DB_QUERY_TIMEOUT", 30))

    async def get_pool(self) -> AsyncConnectionPool:
        """Get or create the connection pool with health check"""
        if self._pool is None:
            try:
                self._pool = AsyncConnectionPool(
                    self.connection_string,
                    min_size=self._min_size,
                    max_size=self._max_size,
                    timeout=self._pool_timeout,
                    kwargs={"row_factory": dict_row}
                )
                await self._pool.open()
                logger.info(f"Database pool created: min_size={self._min_size}, max_size={self._max_size}")
                
                # Verify connection
                await self._verify_connection()
            except Exception as e:
                logger.error(f"Failed to create database pool: {e}", exc_info=True)
                self._pool = None
                raise
        
        # Check if pool is still healthy
        try:
            if self._pool.closed:
                logger.warning("Pool was closed, recreating...")
                self._pool = None
                return await self.get_pool()
        except Exception as e:
            logger.error(f"Pool health check failed: {e}")
            self._pool = None
            return await self.get_pool()
        
        return self._pool

    async def _verify_connection(self):
        """Verify database connectivity"""
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await asyncio.wait_for(cur.execute("SELECT 1"), timeout=5)
            logger.info("Database connection verified")
        except Exception as e:
            logger.error(f"Database connection verification failed: {e}")
            raise

    async def init_database(self):
        """Initialize the ai_test2 table with proper error handling"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                pool = await self.get_pool()
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        # Set statement timeout for DDL operations
                        await cur.execute(f"SET statement_timeout = '{self._query_timeout * 1000}'")
                        
                        # Create ai_test2 table with user_id, chat_id, messages, summary, and image_count
                        await cur.execute("""
                            CREATE TABLE IF NOT EXISTS ai_test2 (
                                id SERIAL PRIMARY KEY,
                                user_id VARCHAR(255) NOT NULL,
                                chat_id VARCHAR(255) NOT NULL,
                                messages JSONB DEFAULT '[]'::jsonb,
                                summary_context TEXT DEFAULT NULL,
                                image_count INTEGER DEFAULT 0,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                UNIQUE(user_id, chat_id)
                            );
                        """)
                        
                        # Create indexes for faster lookups
                        await cur.execute("""
                            CREATE INDEX IF NOT EXISTS idx_user_chat ON ai_test2(user_id, chat_id);
                        """)
                        await cur.execute("""
                            CREATE INDEX IF NOT EXISTS idx_user_id ON ai_test2(user_id);
                        """)
                        await cur.execute("""
                            CREATE INDEX IF NOT EXISTS idx_updated_at ON ai_test2(updated_at DESC);
                        """)
                        
                        await conn.commit()
                        logger.info("Database ai_test2 table initialized successfully")
                        return
                        
            except (OperationalError, InterfaceError) as e:
                logger.warning(f"Database connection error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise
            except Exception as e:
                logger.error(f"Database initialization error: {e}", exc_info=True)
                raise

    async def get_and_increment_image_count(self, user_id: str, chat_id: str) -> int:
        """Get current image count and increment it for next image"""
        if not user_id or not chat_id:
            raise ValueError("user_id and chat_id are required")
        
        try:
            pool = await self.get_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(f"SET statement_timeout = '{self._query_timeout * 1000}'")
                    
                    # Get current count and increment atomically
                    await cur.execute("""
                        INSERT INTO ai_test2 (user_id, chat_id, image_count)
                        VALUES (%s, %s, 1)
                        ON CONFLICT (user_id, chat_id) 
                        DO UPDATE SET 
                            image_count = ai_test2.image_count + 1,
                            updated_at = CURRENT_TIMESTAMP
                        RETURNING image_count
                    """, (user_id, chat_id))
                    
                    result = await cur.fetchone()
                    await conn.commit()
                    
                    current_count = result['image_count'] if result else 1
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
            pool = await self.get_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(f"SET statement_timeout = '{self._query_timeout * 1000}'")
                    
                    await cur.execute(
                        """SELECT user_id, chat_id, messages, summary_context, created_at, updated_at 
                           FROM ai_test2 WHERE user_id = %s AND chat_id = %s""",
                        (user_id, chat_id)
                    )
                    result = await cur.fetchone()
                    return result
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
            pool = await self.get_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(f"SET statement_timeout = '{self._query_timeout * 1000}'")
                    
                    message = {
                        "role": role,
                        "content": content,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    # Add image URL if provided
                    if image_url:
                        message["image_url"] = image_url
                    
                    # Insert or update: append message to messages array
                    await cur.execute("""
                        INSERT INTO ai_test2 (user_id, chat_id, messages, updated_at)
                        VALUES (%s, %s, %s::jsonb, CURRENT_TIMESTAMP)
                        ON CONFLICT (user_id, chat_id) 
                        DO UPDATE SET 
                            messages = ai_test2.messages || %s::jsonb,
                            updated_at = CURRENT_TIMESTAMP
                    """, (user_id, chat_id, json.dumps([message]), json.dumps([message])))
                    
                    await conn.commit()
                    logger.info(f"Message saved for user_id={user_id}, chat_id={chat_id}, role={role}, has_image={image_url is not None}")
        except Exception as e:
            logger.error(f"Error saving message: {e}", exc_info=True)
            raise

    async def save_summary_context(self, user_id: str, chat_id: str, summary_text: str):
        """Save summarized context when chat ends or context gets too large"""
        if not user_id or not chat_id or not summary_text:
            raise ValueError("user_id, chat_id, and summary_text are required")
        
        try:
            pool = await self.get_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(f"SET statement_timeout = '{self._query_timeout * 1000}'")
                    
                    await cur.execute("""
                        UPDATE ai_test2 
                        SET summary_context = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = %s AND chat_id = %s
                    """, (summary_text, user_id, chat_id))
                    
                    await conn.commit()
                    logger.info(f"Summary context saved for user_id={user_id}, chat_id={chat_id}")
        except Exception as e:
            logger.error(f"Error saving summary: {e}", exc_info=True)
            raise

    async def get_user_chats(self, user_id: str) -> List[Dict]:
        """Get all chats for a user"""
        if not user_id:
            raise ValueError("user_id is required")
        
        try:
            pool = await self.get_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(f"SET statement_timeout = '{self._query_timeout * 1000}'")
                    
                    await cur.execute(
                        """SELECT chat_id, created_at, updated_at, 
                           jsonb_array_length(messages) as message_count
                           FROM ai_test2 WHERE user_id = %s ORDER BY updated_at DESC""",
                        (user_id,)
                    )
                    results = await cur.fetchall()
                    return results
        except Exception as e:
            logger.error(f"Error fetching user chats: {e}", exc_info=True)
            raise

    async def delete_chat(self, user_id: str, chat_id: str) -> bool:
        """Delete a specific chat"""
        if not user_id or not chat_id:
            raise ValueError("user_id and chat_id are required")
        
        try:
            pool = await self.get_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(f"SET statement_timeout = '{self._query_timeout * 1000}'")
                    
                    await cur.execute(
                        "DELETE FROM ai_test2 WHERE user_id = %s AND chat_id = %s",
                        (user_id, chat_id)
                    )
                    rows_deleted = cur.rowcount
                    await conn.commit()
                    
                    logger.info(f"Chat deleted: user_id={user_id}, chat_id={chat_id}, rows_deleted={rows_deleted}")
                    return rows_deleted > 0
        except Exception as e:
            logger.error(f"Error deleting chat: {e}", exc_info=True)
            raise

    async def close(self):
        """Close the connection pool gracefully"""
        if self._pool:
            try:
                await self._pool.close()
                self._pool = None
                logger.info("Database connection pool closed successfully")
            except Exception as e:
                logger.error(f"Error closing database pool: {e}", exc_info=True)

# Singleton instance
db = Database()
