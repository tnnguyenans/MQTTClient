"""Configuration management for MQTT Client application."""

import os
from pathlib import Path
from typing import Optional, Tuple, Literal, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class MQTTConfig(BaseModel):
    """MQTT broker configuration."""
    
    broker_host: str = Field(default="localhost", description="MQTT broker hostname")
    broker_port: int = Field(default=1883, description="MQTT broker port")
    client_id: str = Field(default="mqtt_client", description="MQTT client identifier")
    username: Optional[str] = Field(default=None, description="MQTT broker username")
    password: Optional[str] = Field(default=None, description="MQTT broker password")
    topic: str = Field(default="anstest", description="MQTT topic to subscribe to")
    qos: int = Field(default=0, description="Quality of Service level (0, 1, or 2)")
    clean_session: bool = Field(default=True, description="Whether to start with a clean session")


class ImageProcessorConfig(BaseModel):
    """Image processor configuration."""
    
    cache_dir: str = Field(
        default=os.path.join(str(Path.home()), ".mqtt_client_cache"),
        description="Directory to store cached images"
    )
    max_cache_size_mb: int = Field(
        default=500,
        description="Maximum cache size in megabytes"
    )
    default_max_size: Tuple[int, int] = Field(
        default=(1280, 720),
        description="Default maximum image size (width, height)"
    )
    jpeg_quality: int = Field(
        default=85,
        description="JPEG quality for saved images (0-100)"
    )


class LLMConfig(BaseModel):
    """LLM configuration for image analysis."""
    
    provider: Literal["openai", "ollama"] = Field(
        default="openai",
        description="LLM provider (openai or ollama)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for OpenAI"
    )
    model: str = Field(
        default="gpt-4o",
        description="Model to use for OpenAI or Ollama"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama API"
    )
    max_tokens: int = Field(
        default=300,
        description="Maximum number of tokens in response"
    )
    temperature: float = Field(
        default=0.7,
        description="Temperature for response generation (0.0-1.0)"
    )
    timeout: int = Field(
        default=30,
        description="Timeout for API requests in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed requests"
    )
    rate_limit: int = Field(
        default=10,
        description="Maximum number of requests per minute"
    )


class APIConfig(BaseModel):
    """API configuration."""
    
    host: str = Field(
        default="0.0.0.0",
        description="Host address for the API server"
    )
    port: int = Field(
        default=8000,
        description="Port for the API server"
    )
    cors_origins: List[str] = Field(
        default=["*"],
        description="List of allowed CORS origins"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode for API"
    )
    max_history_size: int = Field(
        default=100,
        description="Maximum number of detection events to keep in history"
    )


class AppConfig(BaseModel):
    """Application configuration."""
    
    version: str = Field(default="1.0.0", description="Application version")
    mqtt: MQTTConfig = Field(default_factory=MQTTConfig, description="MQTT configuration")
    image_processor: ImageProcessorConfig = Field(
        default_factory=ImageProcessorConfig,
        description="Image processor configuration"
    )
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM configuration"
    )
    api: APIConfig = Field(
        default_factory=APIConfig,
        description="API configuration"
    )
    debug: bool = Field(default=False, description="Enable debug mode")


def load_config() -> AppConfig:
    """Load application configuration.
    
    Returns:
        AppConfig: Application configuration object.
    """
    # Load .env file from different possible locations
    for env_path in [
        os.path.join(os.getcwd(), '.env'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'),
    ]:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"Loaded environment variables from {env_path}")
            break
    
    # Create default config
    config = AppConfig()
    
    # Load from environment variables if available
    # LLM configuration
    if os.environ.get("OPENAI_API_KEY"):
        config.llm.api_key = os.environ.get("OPENAI_API_KEY")
    
    if os.environ.get("LLM_PROVIDER"):
        provider = os.environ.get("LLM_PROVIDER").lower()
        if provider in ["openai", "ollama"]:
            config.llm.provider = provider
    
    if os.environ.get("LLM_MODEL"):
        config.llm.model = os.environ.get("LLM_MODEL")
    
    if os.environ.get("OLLAMA_BASE_URL"):
        config.llm.ollama_base_url = os.environ.get("OLLAMA_BASE_URL")
    
    # API configuration
    if os.environ.get("API_HOST"):
        config.api.host = os.environ.get("API_HOST")
    
    if os.environ.get("API_PORT") and os.environ.get("API_PORT").isdigit():
        config.api.port = int(os.environ.get("API_PORT"))
    
    if os.environ.get("API_DEBUG") and os.environ.get("API_DEBUG").lower() in ["true", "1", "yes"]:
        config.api.debug = True
    
    # Create cache directory if it doesn't exist
    os.makedirs(config.image_processor.cache_dir, exist_ok=True)
    
    return config
