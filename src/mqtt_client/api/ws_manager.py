"""WebSocket connection manager for FastAPI."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect

# Configure logging
logger = logging.getLogger(__name__)


class ConnectionManager:
    """WebSocket connection manager for handling multiple connections."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Set[WebSocket] = set()
        self.connection_count: int = 0
        logger.info("WebSocket connection manager initialized")
    
    async def connect(self, websocket: WebSocket) -> int:
        """Accept a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection to accept.
            
        Returns:
            int: Client ID assigned to the connection.
        """
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_count += 1
        client_id = self.connection_count
        
        logger.info(f"Client {client_id} connected (Total connections: {len(self.active_connections)})")
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection_established",
            "data": {
                "client_id": client_id,
                "message": "Connection established"
            }
        }, websocket)
        
        return client_id
    
    def disconnect(self, websocket: WebSocket) -> None:
        """Handle WebSocket disconnection.
        
        Args:
            websocket: WebSocket connection to disconnect.
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Client disconnected (Remaining connections: {len(self.active_connections)})")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket) -> None:
        """Send a message to a specific client.
        
        Args:
            message: Message to send.
            websocket: WebSocket connection to send message to.
        """
        try:
            # Add timestamp if not present
            if "timestamp" not in message:
                message["timestamp"] = datetime.utcnow().isoformat()
                
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            # Remove connection if there was an error
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected clients.
        
        Args:
            message: Message to broadcast.
        """
        if not self.active_connections:
            logger.debug("No active connections for broadcast")
            return
        
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()
        
        # Create a copy to prevent modification during iteration
        connections = self.active_connections.copy()
        
        logger.debug(f"Broadcasting message to {len(connections)} clients")
        
        # Send message to all connections
        disconnected = set()
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


# Create a global connection manager instance
connection_manager = ConnectionManager()