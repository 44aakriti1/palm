"""Redis chat memory service."""
import json
from typing import List, Dict, Any
import redis.asyncio as redis
from app.core.config import get_settings

settings = get_settings()


class ChatMemoryService:
    """Service for managing chat history in Redis."""
    
    def __init__(self) -> None:
        """Initialize Redis connection."""
        self.redis_client: redis.Redis | None = None
    
    async def _get_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self.redis_client is None:
            self.redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        return self.redis_client
    
    def _get_key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"chat:{session_id}"
    
    async def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str,
        metadata: Dict[str, Any] | None = None
    ) -> None:
        """Add a message to chat history.
        
        Args:
            session_id: Unique session identifier
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional additional data
        """
        client = await self._get_client()
        key = self._get_key(session_id)
        
        message = {
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        
        await client.rpush(key, json.dumps(message))
        await client.expire(key, 86400)  # 24 hour TTL
    
    async def get_history(
        self, 
        session_id: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get chat history for a session.
        
        Args:
            session_id: Unique session identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of messages
        """
        client = await self._get_client()
        key = self._get_key(session_id)
        
        messages = await client.lrange(key, -limit, -1)
        return [json.loads(m) for m in messages]
    
    async def clear_history(self, session_id: str) -> None:
        """Clear chat history for a session.
        
        Args:
            session_id: Unique session identifier
        """
        client = await self._get_client()
        key = self._get_key(session_id)
        await client.delete(key)
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
