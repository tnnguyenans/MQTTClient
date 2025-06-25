"""Data transformation utilities for MQTT client data models."""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from mqtt_client.models.detection import DetectionData, BoundingBox, DetectedObject
from mqtt_client.models.alpr_detection import ALPREventData, DetectionItemModel

# Configure logging
logger = logging.getLogger(__name__)


def extract_bounding_boxes(
    event_data: ALPREventData
) -> List[Dict[str, Any]]:
    """Extract bounding boxes from ALPR event data.
    
    Args:
        event_data: Validated ALPR event data.
        
    Returns:
        List of dictionaries with bounding box information.
    """
    boxes = []
    
    # Check if Detections is present and not empty
    if not event_data.Detections:
        logger.warning("No Detections found in ALPR event data")
        return boxes
    
    try:
        for rule in event_data.Detections:
            # Check if DetectionData is present and not empty
            if not rule.DetectionData:
                logger.warning("No DetectionData found in rule")
                continue
                
            for detection_data in rule.DetectionData:
                # Check if Detections is present and not empty
                if not detection_data.Detections:
                    logger.warning(f"No Detections found in detection data {detection_data.Name}")
                    continue
                    
                for detection in detection_data.Detections:
                    # Check if BoundingBox is present
                    if not hasattr(detection, 'BoundingBox') or not detection.BoundingBox:
                        logger.warning(f"No BoundingBox found in detection for {detection_data.Name}")
                        continue
                        
                    bbox = detection.BoundingBox
                    
                    # Convert bounding box to the format expected by the analyzer
                    box_info = {
                        'license_plate': detection_data.Name,
                        'confidence': detection.Score,
                        'timestamp': detection.Time,
                        'bounding_box': {
                            'x': bbox.Left,
                            'y': bbox.Top,
                            'width': bbox.Right - bbox.Left,
                            'height': bbox.Bottom - bbox.Top
                        },
                        'camera_id': detection_data.CameraID,
                        'camera_name': detection_data.CameraName
                    }
                    
                    boxes.append(box_info)
    except Exception as e:
        logger.error(f"Error extracting bounding boxes: {e}")
    
    return boxes


def map_alpr_to_detection_data(
    event_data: ALPREventData
) -> DetectionData:
    """Map ALPR event data to generic detection data format.
    
    This allows for consistent processing regardless of input format.
    
    Args:
        event_data: Validated ALPR event data.
        
    Returns:
        Mapped DetectionData object.
    """
    # Extract timestamp from first detection if available
    timestamp = datetime.now()
    if event_data.Detections and event_data.Detections[0].DetectionData:
        first_detection = event_data.Detections[0].DetectionData[0]
        if first_detection.Detections:
            timestamp = first_detection.Detections[0].Time
    
    # Create detected objects from ALPR detections
    objects = []
    for rule in event_data.Detections:
        for detection_data in rule.DetectionData:
            for detection in detection_data.Detections:
                bbox = detection.BoundingBox
                
                # Create bounding box
                bounding_box = BoundingBox(
                    x=bbox.Left,
                    y=bbox.Top,
                    width=bbox.Right - bbox.Left,
                    height=bbox.Bottom - bbox.Top
                )
                
                # Create detected object
                obj = DetectedObject(
                    class_id=detection.ModelClassID.Class,
                    class_name=detection_data.Name,  # Use license plate as class name
                    confidence=detection.Score,
                    bounding_box=bounding_box
                )
                objects.append(obj)
    
    # Map image URLs
    image_urls = [str(image.Image) for image in event_data.Images]
    
    # Create detection data
    return DetectionData(
        timestamp=timestamp,
        source=f"{event_data.CameraName} ({event_data.CameraID})",
        objects=objects,
        image_url=image_urls[0] if image_urls else None,
        additional_image_urls=image_urls[1:] if len(image_urls) > 1 else []
    )


def enrich_detection_data(
    detection_data: DetectionData,
    metadata: Optional[Dict[str, Any]] = None
) -> DetectionData:
    """Enrich detection data with additional metadata.
    
    Args:
        detection_data: Detection data to enrich.
        metadata: Additional metadata to add.
        
    Returns:
        Enriched detection data.
    """
    # Create a copy to avoid modifying the original
    enriched = detection_data.copy(deep=True)
    
    # Add metadata if provided
    if metadata:
        # Convert metadata to proper format if needed
        formatted_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool, list, dict)):
                formatted_metadata[key] = value
            else:
                # Convert non-serializable types to string
                formatted_metadata[key] = str(value)
        
        # Update metadata
        if hasattr(enriched, 'metadata') and enriched.metadata:
            enriched.metadata.update(formatted_metadata)
        else:
            enriched.metadata = formatted_metadata
    
    return enriched


def normalize_confidence_scores(
    detection_data: Union[DetectionData, ALPREventData]
) -> Union[DetectionData, ALPREventData]:
    """Normalize confidence scores to range [0.0, 1.0].
    
    Args:
        detection_data: Detection data to normalize.
        
    Returns:
        Detection data with normalized confidence scores.
    """
    if isinstance(detection_data, DetectionData):
        # Handle DetectionData
        for obj in detection_data.objects:
            if obj.confidence > 1.0:
                obj.confidence = obj.confidence / 100.0
            elif obj.confidence < 0.0:
                obj.confidence = 0.0
            elif obj.confidence > 1.0:
                obj.confidence = 1.0
    
    elif isinstance(detection_data, ALPREventData):
        # Handle ALPREventData
        for rule in detection_data.Detections:
            for detection_data_item in rule.DetectionData:
                for detection in detection_data_item.Detections:
                    if detection.Score > 1.0:
                        detection.Score = detection.Score / 100.0
                    elif detection.Score < 0.0:
                        detection.Score = 0.0
                    elif detection.Score > 1.0:
                        detection.Score = 1.0
    
    return detection_data
