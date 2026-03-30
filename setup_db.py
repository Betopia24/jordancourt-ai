import os
import asyncio
from dotenv import load_dotenv
import redis.asyncio as redis

load_dotenv()

async def setup_database():
    connection_string = os.getenv("REDIS_URL")
    if not connection_string:
        raise ValueError("REDIS_URL not found in .env")
    
    client = redis.from_url(connection_string, decode_responses=True)
    try:
        pong = await client.ping()
        if not pong:
            raise RuntimeError("Redis ping failed")
        print("Redis setup complete.")
    finally:
        await client.aclose()

if __name__ == "__main__":
    asyncio.run(setup_database())
