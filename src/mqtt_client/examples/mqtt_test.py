"""Example script to test MQTT client functionality."""

import json
import logging
import time
from datetime import datetime

import paho.mqtt.client as mqtt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def publish_test_message(broker_host="localhost", broker_port=1883, topic="detection/objects"):
    """Publish a test detection message to the MQTT broker.
    
    Args:
        broker_host: MQTT broker hostname.
        broker_port: MQTT broker port.
        topic: MQTT topic to publish to.
    """
    # Create client
    client = mqtt.Client(client_id="test_publisher")
    
    try:
        # Connect to broker
        logger.info(f"Connecting to MQTT broker at {broker_host}:{broker_port}")
        client.connect(broker_host, broker_port)
        client.loop_start()
        
        # Create test detection data
        detection_data = {
            "detection_id": f"test-{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "image_url": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
            "source": "test-camera",
            "objects": [
                {
                    "object_id": "obj-1",
                    "class_name": "person",
                    "confidence": 0.95,
                    "bbox": {
                        "x": 10.0,
                        "y": 20.0,
                        "width": 100.0,
                        "height": 200.0
                    }
                },
                {
                    "object_id": "obj-2",
                    "class_name": "car",
                    "confidence": 0.87,
                    "bbox": {
                        "x": 150.0,
                        "y": 120.0,
                        "width": 80.0,
                        "height": 60.0
                    }
                }
            ]
        }
        
        # Convert to JSON and publish
        payload = json.dumps(detection_data)
        logger.info(f"Publishing test message to topic: {topic}")
        result = client.publish(topic, payload, qos=0)
        
        # Check if publish was successful
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logger.info("Test message published successfully")
        else:
            logger.error(f"Failed to publish test message: {result}")
            
        # Wait a moment for message to be delivered
        time.sleep(1)
        
    except Exception as e:
        logger.error(f"Error publishing test message: {e}")
    finally:
        # Disconnect client
        client.loop_stop()
        client.disconnect()
        logger.info("Disconnected from MQTT broker")


if __name__ == "__main__":
    publish_test_message()
    logger.info("Test complete")
