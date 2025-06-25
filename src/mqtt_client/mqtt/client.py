"""MQTT client implementation."""

import json
import logging
from typing import Callable, Dict, Optional, Any, Union

import paho.mqtt.client as mqtt
from pydantic import ValidationError

from mqtt_client.config import MQTTConfig
from mqtt_client.models.detection import DetectionData
from mqtt_client.models.alpr_detection import ALPREventData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MQTTClient:
    """MQTT client for subscribing to detection data."""
    
    def __init__(self, config: MQTTConfig):
        """Initialize MQTT client.
        
        Args:
            config: MQTT broker configuration.
        """
        self.config = config
        self.client = mqtt.Client(
            client_id=config.client_id,
            clean_session=config.clean_session
        )
        
        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        # Set authentication if provided
        if config.username and config.password:
            self.client.username_pw_set(config.username, config.password)
            
        # Message callback handler
        self._message_callback: Optional[Callable[[Union[DetectionData, ALPREventData]], None]] = None
        
    def connect(self) -> None:
        """Connect to MQTT broker."""
        try:
            logger.info(f"Connecting to MQTT broker at {self.config.broker_host}:{self.config.broker_port}")
            self.client.connect(self.config.broker_host, self.config.broker_port)
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        self.client.disconnect()
        logger.info("Disconnected from MQTT broker")
    
    def start(self) -> None:
        """Start the MQTT client loop."""
        self.client.loop_start()
        logger.info("MQTT client loop started")
    
    def stop(self) -> None:
        """Stop the MQTT client loop."""
        self.client.loop_stop()
        logger.info("MQTT client loop stopped")
    
    def set_message_callback(self, callback: Callable[[Union[DetectionData, ALPREventData]], None]) -> None:
        """Set callback for handling validated detection data.
        
        Args:
            callback: Function to call with validated detection data (DetectionData or ALPREventData).
        """
        self._message_callback = callback
    
    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: Dict, rc: int) -> None:
        """Callback for when client connects to broker.
        
        Args:
            client: MQTT client instance.
            userdata: User data supplied to client.
            flags: Response flags from broker.
            rc: Connection result code.
        """
        if rc == 0:
            logger.info(f"Connected to MQTT broker with result code {rc}")
            # Subscribe to topic upon successful connection
            client.subscribe(self.config.topic, qos=self.config.qos)
            logger.info(f"Subscribed to topic: {self.config.topic}")
        else:
            logger.error(f"Failed to connect to MQTT broker with result code {rc}")
    
    def _on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        """Callback for when a message is received from broker.
        
        Args:
            client: MQTT client instance.
            userdata: User data supplied to client.
            msg: Received message.
        """
        try:
            logger.debug(f"Received message on topic {msg.topic}")
            # Parse JSON payload
            payload = json.loads(msg.payload.decode('utf-8'))
            
            # Try to determine the data format and validate with appropriate Pydantic model
            try:
                # First try to parse as ALPR event data
                if "EventName" in payload and "CameraID" in payload and "Detections" in payload:
                    event_data = ALPREventData.parse_obj(payload)
                    logger.info("Received ALPR event data")
                    
                    # Call user-provided callback if set
                    if self._message_callback:
                        self._message_callback(event_data)
                else:
                    # Fall back to generic detection data format
                    detection_data = DetectionData.parse_obj(payload)
                    logger.info("Received generic detection data")
                    
                    # Call user-provided callback if set
                    if self._message_callback:
                        self._message_callback(detection_data)
            except ValidationError as e:
                logger.error(f"Invalid data format: {e}")
                # Try the other format as a fallback
                try:
                    if "EventName" in payload:
                        detection_data = DetectionData.parse_obj(payload)
                    else:
                        event_data = ALPREventData.parse_obj(payload)
                        
                    # Call user-provided callback if set
                    if self._message_callback:
                        if "EventName" in payload:
                            self._message_callback(detection_data)
                        else:
                            self._message_callback(event_data)
                except ValidationError:
                    # Both formats failed, re-raise the original error
                    logger.error("Failed to parse message with both data models")
                    raise
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON payload: {e}")
        except ValidationError as e:
            logger.error(f"Invalid detection data format: {e}")
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def _on_disconnect(self, client: mqtt.Client, userdata: Any, rc: int) -> None:
        """Callback for when client disconnects from broker.
        
        Args:
            client: MQTT client instance.
            userdata: User data supplied to client.
            rc: Disconnection result code.
        """
        if rc != 0:
            logger.warning(f"Unexpected disconnection from MQTT broker with code {rc}")
        else:
            logger.info("Disconnected from MQTT broker")
