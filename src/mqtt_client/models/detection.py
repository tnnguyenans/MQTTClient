"""Detection data models for MQTT Client application."""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl


class BoundingBox(BaseModel):
    """Bounding box coordinates for detected objects."""
    
    x: float = Field(..., description="X coordinate of top-left corner")
    y: float = Field(..., description="Y coordinate of top-left corner")
    width: float = Field(..., description="Width of bounding box")
    height: float = Field(..., description="Height of bounding box")


class DetectedObject(BaseModel):
    """Detected object information."""
    
    class_id: int = Field(..., description="Class ID of the detected object")
    class_name: str = Field(..., description="Class name of the detected object")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    bounding_box: BoundingBox = Field(..., description="Bounding box coordinates")
    object_id: Optional[str] = Field(None, description="Optional unique identifier for the detected object")


class DetectionData(BaseModel):
    """Detection data received from MQTT broker."""
    
    timestamp: datetime = Field(default_factory=datetime.now, description="Detection timestamp")
    source: str = Field(..., description="Source of the detection (camera ID, etc.)")
    objects: List[DetectedObject] = Field(default_factory=list, description="List of detected objects")
    image_url: Optional[HttpUrl] = Field(None, description="URL to the detected image")
    additional_image_urls: List[HttpUrl] = Field(default_factory=list, description="Additional image URLs")
    detection_id: Optional[str] = Field(None, description="Optional unique identifier for the detection")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata")
