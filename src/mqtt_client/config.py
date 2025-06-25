"""Configuration management for MQTT Client application."""

import os
from pathlib import Path
from typing import Optional, Tuple
from pydantic import BaseModel, Field


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


class AppConfig(BaseModel):
    """Application configuration."""
    
    mqtt: MQTTConfig = Field(default_factory=MQTTConfig, description="MQTT configuration")
    image_processor: ImageProcessorConfig = Field(
        default_factory=ImageProcessorConfig,
        description="Image processor configuration"
    )
    debug: bool = Field(default=False, description="Enable debug mode")


def load_config() -> AppConfig:
    """Load application configuration.
    
    Returns:
        AppConfig: Application configuration object.
    """
    # In a future enhancement, this could load from environment variables or config file
    config = AppConfig()
    
    # Create cache directory if it doesn't exist
    os.makedirs(config.image_processor.cache_dir, exist_ok=True)
    
    return config
