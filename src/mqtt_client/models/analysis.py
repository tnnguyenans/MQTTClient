"""Analysis data models for LLM image analysis."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ObjectAnalysis(BaseModel):
    """Analysis results for a detected object."""
    
    colors: List[str] = Field(default_factory=list, description="List of colors detected in the object")
    actions: List[str] = Field(default_factory=list, description="List of actions or states detected")
    description: str = Field("", description="Brief description of the object")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Additional attributes detected")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")


class AnalysisResult(BaseModel):
    """Complete analysis result for a detection."""
    
    detection_id: Optional[str] = Field(None, description="ID of the detection")
    source: str = Field(..., description="Source of the detection (camera ID, etc.)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    object_analyses: Dict[str, ObjectAnalysis] = Field(
        default_factory=dict, 
        description="Dictionary mapping object IDs to their analyses"
    )
    raw_llm_response: Optional[str] = Field(None, description="Raw LLM response for debugging")
    processing_time_ms: float = Field(0.0, description="Processing time in milliseconds")
    model_used: str = Field("", description="LLM model used for analysis")
    provider: str = Field("", description="LLM provider used (openai or ollama)")