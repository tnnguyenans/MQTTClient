"""Example script to test MQTT client with ALPR data format."""

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


def publish_alpr_test_message(broker_host="localhost", broker_port=1883, topic="anstest"):
    """Publish a test ALPR detection message to the MQTT broker.
    
    Args:
        broker_host: MQTT broker hostname.
        broker_port: MQTT broker port.
        topic: MQTT topic to publish to.
    """
    # Create client
    client = mqtt.Client(client_id="alpr_test_publisher")
    
    try:
        # Connect to broker
        logger.info(f"Connecting to MQTT broker at {broker_host}:{broker_port}")
        client.connect(broker_host, broker_port)
        client.loop_start()
        
        # Create test ALPR event data
        timestamp = datetime.now().isoformat()
        image_id = int(time.time())
        
        alpr_data = {
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
                                    "Time": timestamp,
                                    "BoundingBox": {"Left": 683, "Top": 611, "Right": 727, "Bottom": 643},
                                    "Score": 0.836019754,
                                    "Image": image_id,
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
                    "ID": image_id,
                    "Image": "https://anspushnotification.s3.ap-southeast-2.amazonaws.com/aiboxevents/20250625_122406_1973760195_798",
                    "ScaledFactor": 1
                }
            ]
        }
        
        # Convert to JSON and publish
        payload = json.dumps(alpr_data)
        logger.info(f"Publishing ALPR test message to topic: {topic}")
        result = client.publish(topic, payload, qos=0)
        
        # Check if publish was successful
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logger.info("ALPR test message published successfully")
        else:
            logger.error(f"Failed to publish ALPR test message: {result}")
            
        # Wait a moment for message to be delivered
        time.sleep(1)
        
    except Exception as e:
        logger.error(f"Error publishing ALPR test message: {e}")
    finally:
        # Disconnect client
        client.loop_stop()
        client.disconnect()
        logger.info("Disconnected from MQTT broker")


if __name__ == "__main__":
    publish_alpr_test_message()
    logger.info("ALPR test complete")
