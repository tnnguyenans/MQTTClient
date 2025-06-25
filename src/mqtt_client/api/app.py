"""FastAPI application for MQTT Client."""

import logging
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from mqtt_client.config import load_config
from mqtt_client.api.routes import router

# Configure logging
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application.
    """
    # Load configuration
    config = load_config()
    
    # Create FastAPI app
    app = FastAPI(
        title="MQTT Client API",
        description="API for MQTT Client with LLM Image Analysis",
        version=config.version,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For development; restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Import os and Path for file operations
    import os
    from pathlib import Path
    
    # Determine the static files directory path
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    
    # Check if directory exists
    if not os.path.exists(static_dir):
        logger.warning(f"Static directory not found at {static_dir}, creating it")
        os.makedirs(static_dir, exist_ok=True)
    
    logger.info(f"Mounting static files from: {static_dir}")
    
    # Define routes for HTML files
    @app.get("/")
    async def serve_index():
        """Serve the index.html file."""
        return FileResponse(os.path.join(static_dir, "index.html"))
    
    @app.get("/index.html")
    async def serve_index_html():
        """Serve the index.html file explicitly."""
        return FileResponse(os.path.join(static_dir, "index.html"))
    
    @app.get("/analytics.html")
    async def serve_analytics():
        """Serve the analytics.html file."""
        return FileResponse(os.path.join(static_dir, "analytics.html"))
    
    # Mount static files for CSS, JS, etc.
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # Mount API router directly at root path for API endpoints
    app.include_router(router)
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle all unhandled exceptions.
        
        Args:
            request: FastAPI request.
            exc: Exception.
            
        Returns:
            JSONResponse: Error response.
        """
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    # Add startup and shutdown events
    @app.on_event("startup")
    async def startup_event() -> None:
        """Handle application startup."""
        logger.info("Starting FastAPI application")
    
    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        """Handle application shutdown."""
        logger.info("Shutting down FastAPI application")
    
    # Return configured app
    return app