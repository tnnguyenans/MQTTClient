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
    
    object_id: str = Field(..., description="Unique identifier for the detected object")
    class_name: str = Field(..., description="Class name of the detected object")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")


class DetectionData(BaseModel):
    """Detection data received from MQTT broker."""
    
    detection_id: str = Field(..., description="Unique identifier for the detection")
    timestamp: datetime = Field(default_factory=datetime.now, description="Detection timestamp")
    image_url: HttpUrl = Field(..., description="URL to the detected image")
    source: str = Field(..., description="Source of the detection (camera ID, etc.)")
    objects: List[DetectedObject] = Field(default=[], description="List of detected objects")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata")
