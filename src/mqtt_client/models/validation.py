"""Validation utilities for MQTT client data models."""

import logging
from typing import Dict, Any, Union, Optional, TypeVar, Type, List, Callable
from datetime import datetime

from pydantic import ValidationError, BaseModel

from mqtt_client.models.detection import DetectionData
from mqtt_client.models.alpr_detection import ALPREventData

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic model validation
ModelT = TypeVar('ModelT', bound=BaseModel)


class ValidationResult:
    """Result of a validation operation."""

    def __init__(
        self, 
        is_valid: bool, 
        model: Optional[BaseModel] = None, 
        errors: Optional[List[Dict[str, Any]]] = None,
        model_type: Optional[str] = None
    ):
        """Initialize validation result.
        
        Args:
            is_valid: Whether the data is valid.
            model: Validated model instance if valid.
            errors: List of validation errors if invalid.
            model_type: String identifier of the model type.
        """
        self.is_valid = is_valid
        self.model = model
        self.errors = errors or []
        self.model_type = model_type

    def __bool__(self) -> bool:
        """Allow using the result in boolean context.
        
        Returns:
            True if validation was successful, False otherwise.
        """
        return self.is_valid


def validate_model(data: Dict[str, Any], model_class: Type[ModelT]) -> ValidationResult:
    """Validate data against a Pydantic model.
    
    Args:
        data: Data to validate.
        model_class: Pydantic model class to validate against.
        
    Returns:
        ValidationResult with validation status and details.
    """
    try:
        model_instance = model_class.parse_obj(data)
        return ValidationResult(
            is_valid=True, 
            model=model_instance,
            model_type=model_class.__name__
        )
    except ValidationError as e:
        errors = e.errors()
        # Format errors for better readability
        formatted_errors = []
        for error in errors:
            formatted_errors.append({
                'location': '.'.join(str(loc) for loc in error['loc']),
                'message': error['msg'],
                'type': error['type']
            })
        
        logger.error(f"Validation error for {model_class.__name__}: {formatted_errors}")
        return ValidationResult(
            is_valid=False, 
            errors=formatted_errors,
            model_type=model_class.__name__
        )


def detect_and_validate_data(data: Dict[str, Any]) -> ValidationResult:
    """Detect data format and validate against appropriate model.
    
    Args:
        data: Data to validate.
        
    Returns:
        ValidationResult with validation status and details.
    """
    # Try to determine the data format based on fields
    if "EventName" in data and "CameraID" in data and "Detections" in data and "Images" in data:
        # ALPR event data format
        return validate_model(data, ALPREventData)
    elif "timestamp" in data and "objects" in data:
        # Generic detection data format
        return validate_model(data, DetectionData)
    else:
        # Try both formats as a fallback
        alpr_result = validate_model(data, ALPREventData)
        if alpr_result.is_valid:
            return alpr_result
            
        detection_result = validate_model(data, DetectionData)
        if detection_result.is_valid:
            return detection_result
            
        # Both formats failed
        logger.error("Data doesn't match any known format")
        return ValidationResult(
            is_valid=False,
            errors=[{
                'location': 'root',
                'message': 'Data does not match any known format (ALPREventData or DetectionData)',
                'type': 'format_error'
            }],
            model_type='unknown'
        )


def transform_datetime_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform string datetime fields to datetime objects.
    
    Args:
        data: Data dictionary to transform.
        
    Returns:
        Transformed data dictionary.
    """
    result = data.copy()
    
    def process_dict(d: Dict[str, Any]) -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                process_dict(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        process_dict(item)
            elif isinstance(value, str) and key.lower() in ('time', 'timestamp', 'date'):
                try:
                    # Try to parse as ISO format datetime
                    d[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    # If parsing fails, keep the original value
                    pass
    
    process_dict(result)
    return result


def extract_license_plates(event_data: ALPREventData) -> List[str]:
    """Extract license plate numbers from ALPR event data.
    
    Args:
        event_data: Validated ALPR event data.
        
    Returns:
        List of license plate numbers.
    """
    plates = []
    for rule in event_data.Detections:
        for detection_data in rule.DetectionData:
            if detection_data.Name:
                plates.append(detection_data.Name)
    
    return plates


def extract_image_urls(event_data: ALPREventData) -> List[str]:
    """Extract image URLs from ALPR event data.
    
    Args:
        event_data: Validated ALPR event data.
        
    Returns:
        List of image URLs.
    """
    return [str(image.Image) for image in event_data.Images]
