"""Configuration management for MQTT Client application."""

from typing import Optional
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


class AppConfig(BaseModel):
    """Application configuration."""
    
    mqtt: MQTTConfig = Field(default_factory=MQTTConfig, description="MQTT configuration")
    debug: bool = Field(default=False, description="Enable debug mode")


def load_config() -> AppConfig:
    """Load application configuration.
    
    Returns:
        AppConfig: Application configuration object.
    """
    # In a future enhancement, this could load from environment variables or config file
    return AppConfig()
