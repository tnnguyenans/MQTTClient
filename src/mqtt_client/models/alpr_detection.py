"""ALPR (Automatic License Plate Recognition) data models."""

from typing import List, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl, validator


class ModelClassIDModel(BaseModel):
    """Model and class identifiers."""
    
    ModelID: int = Field(..., description="Model identifier")
    Class: int = Field(..., description="Class identifier")


class ModelClassNameModel(BaseModel):
    """Model and class names."""    
    Model: str = Field(..., description="Model name")
    Class: str = Field(..., description="Class name")


class BoundingBoxModel(BaseModel):
    """Bounding box coordinates for detected objects.
    
    Supports both uppercase (Left, Top, Right, Bottom) and
    lowercase (left, top, right, bottom) attribute names.
    """
    
    Left: int = Field(0, description="Left coordinate")
    Top: int = Field(0, description="Top coordinate")
    Right: int = Field(0, description="Right coordinate")
    Bottom: int = Field(0, description="Bottom coordinate")
    
    # Allow field aliases for lowercase variants
    class Config:
        allow_population_by_field_name = True
        fields = {
            'Left': {'alias': 'left'},
            'Top': {'alias': 'top'},
            'Right': {'alias': 'right'},
            'Bottom': {'alias': 'bottom'}
        }
        
    @validator('*', pre=True)
    def ensure_coordinates(cls, v, values, **kwargs):
        """Ensure coordinates are valid integers."""
        if v is None:
            return 0
        return v


class UserInfoModel(BaseModel):
    """User information."""
    
    UserID: str = Field(default="", description="User identifier")
    UserName: str = Field(default="", description="User name")
    GroupIDs: List[Any] = Field(default_factory=list, description="Group identifiers")
    GroupNames: List[str] = Field(default_factory=list, description="Group names")


class DetectionItemModel(BaseModel):
    """Individual detection information."""
    
    Time: datetime = Field(..., description="Detection timestamp")
    BoundingBox: BoundingBoxModel = Field(..., description="Bounding box coordinates")
    Score: float = Field(..., description="Confidence score (0.0-1.0)")
    Image: int = Field(..., description="Image identifier")
    ModelClassID: ModelClassIDModel = Field(..., description="Model and class identifiers")
    ModelClassName: ModelClassNameModel = Field(..., description="Model and class names")
    ExtraInformation: str = Field(default="", description="Additional information")
    Attributes: List[Any] = Field(default_factory=list, description="Detection attributes")


class DetectionDataModel(BaseModel):
    """Detection data for a single object."""
    
    Name: str = Field(..., description="Detection name (e.g., license plate number)")
    ModelClassID: ModelClassIDModel = Field(..., description="Model and class identifiers")
    ModelClassName: ModelClassNameModel = Field(..., description="Model and class names")
    DeploymentGroupID: int = Field(..., description="Deployment group identifier")
    DeploymentGroupName: str = Field(..., description="Deployment group name")
    CameraID: int = Field(..., description="Camera identifier")
    CameraName: str = Field(..., description="Camera name")
    CameraAddress: str = Field(..., description="Camera address or file path")
    TrackID: int = Field(..., description="Track identifier")
    Detections: List[DetectionItemModel] = Field(..., description="List of detections")
    Result: str = Field(default="", description="Detection result")
    ExtraVisionResult: str = Field(default="", description="Extra vision result")
    TriggeredGroup: str = Field(default="", description="Triggered group")
    UserInfo: UserInfoModel = Field(default_factory=UserInfoModel, description="User information")


class RuleModel(BaseModel):
    """Rule information."""
    
    RuleType: str = Field(..., description="Type of rule (e.g., 'Presence')")
    DetectionData: List[DetectionDataModel] = Field(..., description="List of detection data")
    PipelineOption: str = Field(..., description="Pipeline option")


class ImageInfoModel(BaseModel):
    """Image information.
    
    The Image field can be either a URL or a base64 encoded string.
    """
    
    ID: int = Field(..., description="Image identifier")
    Image: str = Field(..., description="Image URL or base64 encoded string")
    ScaledFactor: float = Field(default=1.0, description="Image scaling factor")
    
    @validator('Image')
    def validate_image(cls, v):
        """Validate image field as either URL or base64.
        
        Args:
            v: Image field value.
            
        Returns:
            str: Validated image field value.
        """
        # If it starts with http:// or https://, validate as URL
        if v.startswith(('http://', 'https://')):
            # We're not using HttpUrl validator directly to avoid the 2083 character limit
            # Just do basic validation
            if not '://' in v:
                raise ValueError('URL must contain scheme (http:// or https://)')
        # Otherwise, assume it's a base64 string (will be validated during processing)
        return v


class ALPREventData(BaseModel):
    """ALPR event data received from MQTT broker."""
    
    EventName: str = Field(..., description="Event name")
    CameraID: int = Field(..., description="Camera identifier")
    CameraName: str = Field(..., description="Camera name")
    Detections: List[RuleModel] = Field(..., description="List of detection rules")
    Images: List[ImageInfoModel] = Field(..., description="List of images")