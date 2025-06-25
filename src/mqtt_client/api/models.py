"""Response models for the FastAPI endpoints."""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field


class BoundingBoxModel(BaseModel):
    """Bounding box model for API responses."""
    
    x: float = Field(..., description="X coordinate of the bounding box")
    y: float = Field(..., description="Y coordinate of the bounding box")
    width: float = Field(..., description="Width of the bounding box")
    height: float = Field(..., description="Height of the bounding box")


class DetectionObjectModel(BaseModel):
    """Detection object model for API responses."""
    
    id: str = Field(..., description="Object ID")
    class_name: str = Field(..., description="Class name of the detected object")
    class_id: int = Field(..., description="Class ID of the detected object")
    confidence: float = Field(..., description="Confidence score")
    timestamp: datetime = Field(..., description="Detection timestamp")
    bounding_box: BoundingBoxModel = Field(..., description="Bounding box coordinates")
    image_url: Optional[str] = Field(None, description="URL to the object image")
    analysis: Optional[Dict[str, Any]] = Field(None, description="LLM analysis results")


class LicensePlateModel(BaseModel):
    """License plate model for API responses."""
    
    plate_number: str = Field(..., description="License plate number")
    confidence: float = Field(..., description="Confidence score")
    timestamp: datetime = Field(..., description="Detection timestamp")
    bounding_box: BoundingBoxModel = Field(..., description="Bounding box coordinates")
    image_url: Optional[str] = Field(None, description="URL to the license plate image")
    analysis: Optional[Dict[str, Any]] = Field(None, description="LLM analysis results")


class DetectionEventModel(BaseModel):
    """Detection event model for API responses."""
    
    id: str = Field(..., description="Event ID")
    source: str = Field(..., description="Source of the detection")
    timestamp: datetime = Field(..., description="Event timestamp") 
    objects: List[Union[DetectionObjectModel, LicensePlateModel]] = Field(
        ..., description="Detected objects or license plates"
    )
    image_url: Optional[str] = Field(None, description="URL to the main event image")
    additional_image_urls: List[str] = Field(default=[], description="Additional image URLs")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Service uptime in seconds")
    components: Dict[str, Dict[str, Any]] = Field(
        default={}, description="Component statuses"
    )


class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")


class ImageCacheInfo(BaseModel):
    """Image cache information model."""
    
    cache_size: int = Field(..., description="Current cache size in bytes")
    max_cache_size: int = Field(..., description="Maximum cache size in bytes")
    item_count: int = Field(..., description="Number of items in cache")
    directory: str = Field(..., description="Cache directory path")