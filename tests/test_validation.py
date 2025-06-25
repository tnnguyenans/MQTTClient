"""Tests for validation utilities."""

import json
import pytest
from datetime import datetime
from typing import Dict, Any

from mqtt_client.models.validation import (
    validate_model,
    detect_and_validate_data,
    transform_datetime_fields,
    extract_license_plates,
    extract_image_urls,
    ValidationResult
)
from mqtt_client.models.detection import DetectionData
from mqtt_client.models.alpr_detection import ALPREventData


@pytest.fixture
def valid_detection_data() -> Dict[str, Any]:
    """Fixture for valid detection data."""
    return {
        "timestamp": "2025-06-25T02:24:06.288Z",
        "source": "Camera 1",
        "objects": [
            {
                "class_id": 1,
                "class_name": "license_plate",
                "confidence": 0.85,
                "bounding_box": {
                    "x": 100,
                    "y": 200,
                    "width": 50,
                    "height": 20
                }
            }
        ],
        "image_url": "https://example.com/image.jpg"
    }


@pytest.fixture
def valid_alpr_data() -> Dict[str, Any]:
    """Fixture for valid ALPR event data."""
    return {
        "EventName": "ALPR LLM Task",
        "CameraID": 1973760043,
        "CameraName": "ALPR Day",
        "Detections": [
            {
                "RuleType": "Presence",
                "DetectionData": [
                    {
                        "Name": "59A344886",
                        "ModelClassID": {"ModelID": 4294967294, "Class": 1},
                        "ModelClassName": {"Model": "B-IN_License Plate Recognition v1.0", "Class": ""},
                        "DeploymentGroupID": 1973760195,
                        "DeploymentGroupName": "ALPR LLM Task",
                        "CameraID": 1973760043,
                        "CameraName": "ALPR Day",
                        "CameraAddress": "2025-05-12_02-43-24_PMH_ngay.mp4",
                        "TrackID": 1,
                        "Detections": [
                            {
                                "Time": "2025-06-25T02:24:06.288Z",
                                "BoundingBox": {"Left": 683, "Top": 611, "Right": 727, "Bottom": 643},
                                "Score": 0.836019754,
                                "Image": 284931,
                                "ModelClassID": {"ModelID": 4294967294, "Class": 1},
                                "ModelClassName": {"Model": "B-IN_License Plate Recognition v1.0", "Class": "59A344886"},
                                "ExtraInformation": "",
                                "Attributes": []
                            }
                        ],
                        "Result": "",
                        "ExtraVisionResult": "",
                        "TriggeredGroup": "",
                        "UserInfo": {"UserID": "", "UserName": "", "GroupIDs": [], "GroupNames": []}
                    }
                ],
                "PipelineOption": "ALPR pipeline"
            }
        ],
        "Images": [
            {
                "ID": 284931,
                "Image": "https://anspushnotification.s3.ap-southeast-2.amazonaws.com/aiboxevents/20250625_122406_1973760195_798",
                "ScaledFactor": 1
            }
        ]
    }


@pytest.fixture
def invalid_data() -> Dict[str, Any]:
    """Fixture for invalid data."""
    return {
        "timestamp": "invalid-date",
        "objects": "not-a-list",
        "source": 123  # Should be a string
    }


def test_validate_model_valid_detection(valid_detection_data):
    """Test validating valid detection data."""
    result = validate_model(valid_detection_data, DetectionData)
    assert result.is_valid
    assert isinstance(result.model, DetectionData)
    assert result.model_type == "DetectionData"
    assert not result.errors


def test_validate_model_valid_alpr(valid_alpr_data):
    """Test validating valid ALPR data."""
    # First transform datetime fields
    processed_data = transform_datetime_fields(valid_alpr_data)
    result = validate_model(processed_data, ALPREventData)
    assert result.is_valid
    assert isinstance(result.model, ALPREventData)
    assert result.model_type == "ALPREventData"
    assert not result.errors


def test_validate_model_invalid(invalid_data):
    """Test validating invalid data."""
    result = validate_model(invalid_data, DetectionData)
    assert not result.is_valid
    assert result.model is None
    assert len(result.errors) > 0
    assert result.model_type == "DetectionData"


def test_detect_and_validate_data_detection(valid_detection_data):
    """Test auto-detecting and validating detection data."""
    result = detect_and_validate_data(valid_detection_data)
    assert result.is_valid
    assert isinstance(result.model, DetectionData)


def test_detect_and_validate_data_alpr(valid_alpr_data):
    """Test auto-detecting and validating ALPR data."""
    # First transform datetime fields
    processed_data = transform_datetime_fields(valid_alpr_data)
    result = detect_and_validate_data(processed_data)
    assert result.is_valid
    assert isinstance(result.model, ALPREventData)


def test_detect_and_validate_data_invalid(invalid_data):
    """Test auto-detecting and validating invalid data."""
    result = detect_and_validate_data(invalid_data)
    assert not result.is_valid
    assert result.model is None
    assert len(result.errors) > 0


def test_transform_datetime_fields():
    """Test transforming datetime fields."""
    data = {
        "timestamp": "2025-06-25T02:24:06.288Z",
        "nested": {
            "time": "2025-06-25T02:24:06.288Z",
            "other": "not-a-date"
        },
        "list": [
            {"date": "2025-06-25T02:24:06.288Z"},
            {"time": "invalid-date"}
        ]
    }
    
    result = transform_datetime_fields(data)
    
    # Check that valid dates were transformed
    assert isinstance(result["timestamp"], datetime)
    assert isinstance(result["nested"]["time"], datetime)
    assert isinstance(result["list"][0]["date"], datetime)
    
    # Check that invalid dates were left as strings
    assert result["nested"]["other"] == "not-a-date"
    assert result["list"][1]["time"] == "invalid-date"


def test_extract_license_plates(valid_alpr_data):
    """Test extracting license plates from ALPR data."""
    # First transform and validate
    processed_data = transform_datetime_fields(valid_alpr_data)
    validation_result = validate_model(processed_data, ALPREventData)
    assert validation_result.is_valid
    
    plates = extract_license_plates(validation_result.model)
    assert len(plates) == 1
    assert plates[0] == "59A344886"


def test_extract_image_urls(valid_alpr_data):
    """Test extracting image URLs from ALPR data."""
    # First transform and validate
    processed_data = transform_datetime_fields(valid_alpr_data)
    validation_result = validate_model(processed_data, ALPREventData)
    assert validation_result.is_valid
    
    urls = extract_image_urls(validation_result.model)
    assert len(urls) == 1
    assert urls[0] == "https://anspushnotification.s3.ap-southeast-2.amazonaws.com/aiboxevents/20250625_122406_1973760195_798"
