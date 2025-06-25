"""Main entry point for MQTT Client application."""

import logging
import signal
import sys
from typing import Optional, Union

from mqtt_client.config import load_config
from mqtt_client.models.detection import DetectionData
from mqtt_client.models.alpr_detection import ALPREventData
from mqtt_client.models.validation import extract_license_plates, extract_image_urls
from mqtt_client.models.transformers import extract_bounding_boxes, map_alpr_to_detection_data
from mqtt_client.mqtt.client import MQTTClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables
mqtt_client: Optional[MQTTClient] = None


def handle_detection(data: Union[DetectionData, ALPREventData]) -> None:
    """Handle validated detection data.
    
    Args:
        data: Validated detection data (either DetectionData or ALPREventData).
    """
    if isinstance(data, ALPREventData):
        # Handle ALPR event data
        logger.info(f"Received ALPR event: {data.EventName}")
        logger.info(f"Camera: {data.CameraName} (ID: {data.CameraID})")
        
        # Extract license plates
        plates = extract_license_plates(data)
        logger.info(f"Detected license plates: {', '.join(plates)}")
        
        # Extract bounding boxes for all detections
        boxes = extract_bounding_boxes(data)
        logger.info(f"Found {len(boxes)} detections")
        
        # Log detection details
        for i, box in enumerate(boxes):
            logger.info(f"Detection {i+1}:")
            logger.info(f"  - License plate: {box['license_plate']}")
            logger.info(f"  - Confidence: {box['confidence']:.2f}")
            logger.info(f"  - Time: {box['timestamp']}")
            logger.info(f"  - BoundingBox: Left={box['left']}, Top={box['top']}, Right={box['right']}, Bottom={box['bottom']}")
        
        # Process images
        image_urls = extract_image_urls(data)
        for i, url in enumerate(image_urls):
            logger.info(f"Image {i+1}: {url}")
            
        # For future processing, we can convert to common DetectionData format
        # detection_data = map_alpr_to_detection_data(data)
        # This will be used in Feature 3 for image download and processing
        
    elif isinstance(data, DetectionData):
        # Handle generic detection data
        logger.info(f"Received detection data from: {data.source}")
        logger.info(f"Timestamp: {data.timestamp}")
        logger.info(f"Detected {len(data.objects)} objects")
        
        # Log detection details
        for i, obj in enumerate(data.objects):
            logger.info(f"Object {i+1}:")
            logger.info(f"  - Class: {obj.class_name} (ID: {obj.class_id})")
            logger.info(f"  - Confidence: {obj.confidence:.2f}")
            bbox = obj.bounding_box
            logger.info(f"  - BoundingBox: x={bbox.x}, y={bbox.y}, width={bbox.width}, height={bbox.height}")
        
        # Process image
        if data.image_url:
            logger.info(f"Primary image: {data.image_url}")
        
        # Process additional images
        if data.additional_image_urls:
            for i, url in enumerate(data.additional_image_urls):
                logger.info(f"Additional image {i+1}: {url}")
    
    else:
        logger.warning(f"Received unknown data type: {type(data).__name__}")
        return



def handle_exit(signum, frame) -> None:
    """Handle exit signals.
    
    Args:
        signum: Signal number.
        frame: Current stack frame.
    """
    logger.info("Received exit signal, shutting down...")
    if mqtt_client:
        mqtt_client.stop()
        mqtt_client.disconnect()
    sys.exit(0)


def main() -> None:
    """Run the MQTT client application."""
    global mqtt_client
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    try:
        # Load configuration
        config = load_config()
        
        # Create and configure MQTT client
        mqtt_client = MQTTClient(config.mqtt)
        mqtt_client.set_message_callback(handle_detection)
        
        # Connect to broker and start loop
        mqtt_client.connect()
        mqtt_client.start()
        
        logger.info("MQTT client started. Press Ctrl+C to exit.")
        
        # Keep the main thread alive in a cross-platform way
        try:
            # Use a simple loop instead of signal.pause() which is not available on Windows
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        # Clean up resources
        if mqtt_client:
            mqtt_client.stop()
            mqtt_client.disconnect()


if __name__ == "__main__":
    main()
