"""Tests for data transformation utilities."""

import pytest
from datetime import datetime
from typing import Dict, Any

from mqtt_client.models.transformers import (
    extract_bounding_boxes,
    map_alpr_to_detection_data,
    enrich_detection_data,
    normalize_confidence_scores
)
from mqtt_client.models.detection import DetectionData
from mqtt_client.models.alpr_detection import ALPREventData
from mqtt_client.models.validation import validate_model, transform_datetime_fields


@pytest.fixture
def valid_detection_data() -> DetectionData:
    """Fixture for valid detection data."""
    return DetectionData(
        timestamp=datetime.now(),
        source="Camera 1",
        objects=[
            DetectedObject(
                class_id=1,
                class_name="license_plate",
                confidence=0.85,
                bounding_box=BoundingBox(
                    x=100,
                    y=200,
                    width=50,
                    height=20
                )
            )
        ],
        image_url="https://example.com/image.jpg"
    )


@pytest.fixture
def valid_alpr_data() -> ALPREventData:
    """Fixture for valid ALPR event data."""
    data = {
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
    
    # Transform datetime fields and validate
    processed_data = transform_datetime_fields(data)
    validation_result = validate_model(processed_data, ALPREventData)
    assert validation_result.is_valid
    return validation_result.model


def test_extract_bounding_boxes(valid_alpr_data):
    """Test extracting bounding boxes from ALPR data."""
    boxes = extract_bounding_boxes(valid_alpr_data)
    
    assert len(boxes) == 1
    box = boxes[0]
    
    assert box["license_plate"] == "59A344886"
    assert box["confidence"] == 0.836019754
    assert isinstance(box["timestamp"], datetime)
    assert box["left"] == 683
    assert box["top"] == 611
    assert box["right"] == 727
    assert box["bottom"] == 643
    assert box["camera_id"] == 1973760043
    assert box["camera_name"] == "ALPR Day"


def test_map_alpr_to_detection_data(valid_alpr_data):
    """Test mapping ALPR data to detection data."""
    detection_data = map_alpr_to_detection_data(valid_alpr_data)
    
    assert isinstance(detection_data, DetectionData)
    assert detection_data.source == "ALPR Day (1973760043)"
    assert len(detection_data.objects) == 1
    
    obj = detection_data.objects[0]
    assert obj.class_id == 1
    assert obj.class_name == "59A344886"
    assert obj.confidence == 0.836019754
    
    bbox = obj.bounding_box
    assert bbox.x == 683
    assert bbox.y == 611
    assert bbox.width == 44  # Right - Left
    assert bbox.height == 32  # Bottom - Top
    
    assert detection_data.image_url == "https://anspushnotification.s3.ap-southeast-2.amazonaws.com/aiboxevents/20250625_122406_1973760195_798"


def test_enrich_detection_data(valid_detection_data):
    """Test enriching detection data with metadata."""
    metadata = {
        "location": "Test Location",
        "operator": "Test Operator",
        "tags": ["test", "validation"]
    }
    
    enriched = enrich_detection_data(valid_detection_data, metadata)
    
    # Original should not be modified
    assert not hasattr(valid_detection_data, "metadata") or valid_detection_data.metadata != metadata
    
    # Enriched should have metadata
    assert hasattr(enriched, "metadata")
    assert enriched.metadata["location"] == "Test Location"
    assert enriched.metadata["operator"] == "Test Operator"
    assert enriched.metadata["tags"] == ["test", "validation"]


def test_normalize_confidence_scores_detection():
    """Test normalizing confidence scores in detection data."""
    # Create detection data with out-of-range confidence
    detection_data = DetectionData(
        timestamp=datetime.now(),
        source="Camera 1",
        objects=[
            {
                "class_id": 1,
                "class_name": "license_plate",
                "confidence": 95.5,  # Out of range (should be 0-1)
                "bounding_box": {
                    "x": 100,
                    "y": 200,
                    "width": 50,
                    "height": 20
                }
            },
            {
                "class_id": 2,
                "class_name": "vehicle",
                "confidence": -0.5,  # Out of range (should be 0-1)
                "bounding_box": {
                    "x": 50,
                    "y": 100,
                    "width": 200,
                    "height": 150
                }
            }
        ],
        image_url="https://example.com/image.jpg"
    )
    
    normalized = normalize_confidence_scores(detection_data)
    
    # Check that confidence scores were normalized
    assert normalized.objects[0].confidence == 0.955  # 95.5 / 100
    assert normalized.objects[1].confidence == 0.0    # Clamped to 0


def test_normalize_confidence_scores_alpr(valid_alpr_data):
    """Test normalizing confidence scores in ALPR data."""
    # Modify confidence score to be out of range
    for rule in valid_alpr_data.Detections:
        for detection_data in rule.DetectionData:
            for detection in detection_data.Detections:
                detection.Score = 95.5  # Out of range (should be 0-1)
    
    normalized = normalize_confidence_scores(valid_alpr_data)
    
    # Check that confidence scores were normalized
    for rule in normalized.Detections:
        for detection_data in rule.DetectionData:
            for detection in detection_data.Detections:
                assert detection.Score == 0.955  # 95.5 / 100
