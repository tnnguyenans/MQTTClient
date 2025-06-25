"""Main entry point for MQTT Client application."""

import logging
import signal
import sys
from typing import Optional

from mqtt_client.config import load_config
from mqtt_client.models.alpr_detection import ALPREventData
from mqtt_client.mqtt.client import MQTTClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables
mqtt_client: Optional[MQTTClient] = None


def handle_detection(event_data: ALPREventData) -> None:
    """Handle validated ALPR event data.
    
    Args:
        event_data: Validated ALPR event data.
    """
    logger.info(f"Received event: {event_data.EventName}")
    logger.info(f"Camera: {event_data.CameraName} (ID: {event_data.CameraID})")
    
    # Process detections
    for rule in event_data.Detections:
        logger.info(f"Rule Type: {rule.RuleType}")
        
        for detection_data in rule.DetectionData:
            logger.info(f"Detected license plate: {detection_data.Name}")
            logger.info(f"Model: {detection_data.ModelClassName.Model}")
            
            for detection in detection_data.Detections:
                bbox = detection.BoundingBox
                logger.info(f"  - Time: {detection.Time}")
                logger.info(f"  - Score: {detection.Score:.2f}")
                logger.info(f"  - BoundingBox: Left={bbox.Left}, Top={bbox.Top}, Right={bbox.Right}, Bottom={bbox.Bottom}")
    
    # Process images
    for image in event_data.Images:
        logger.info(f"Image ID: {image.ID}, URL: {image.Image}")



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
