"""ALPR (Automatic License Plate Recognition) data models."""

from typing import List, Any
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl


class ModelClassIDModel(BaseModel):
    """Model and class identifiers."""
    
    ModelID: int = Field(..., description="Model identifier")
    Class: int = Field(..., description="Class identifier")


class ModelClassNameModel(BaseModel):
    """Model and class names."""
    
    Model: str = Field(..., description="Model name")
    Class: str = Field(..., description="Class name")


class BoundingBoxModel(BaseModel):
    """Bounding box coordinates for detected objects."""
    
    Left: int = Field(..., description="Left coordinate")
    Top: int = Field(..., description="Top coordinate")
    Right: int = Field(..., description="Right coordinate")
    Bottom: int = Field(..., description="Bottom coordinate")


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


class UserInfoModel(BaseModel):
    """User information."""
    
    UserID: str = Field(default="", description="User identifier")
    UserName: str = Field(default="", description="User name")
    GroupIDs: List[Any] = Field(default_factory=list, description="Group identifiers")
    GroupNames: List[str] = Field(default_factory=list, description="Group names")


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
    """Image information."""
    
    ID: int = Field(..., description="Image identifier")
    Image: HttpUrl = Field(..., description="Image URL")
    ScaledFactor: float = Field(default=1.0, description="Image scaling factor")


class ALPREventData(BaseModel):
    """ALPR event data received from MQTT broker."""
    
    EventName: str = Field(..., description="Event name")
    CameraID: int = Field(..., description="Camera identifier")
    CameraName: str = Field(..., description="Camera name")
    Detections: List[RuleModel] = Field(..., description="List of detection rules")
    Images: List[ImageInfoModel] = Field(..., description="List of images")