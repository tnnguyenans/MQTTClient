"""Tests for MQTT client module."""

import json
import pytest
from unittest.mock import MagicMock, patch

from mqtt_client.config import MQTTConfig
from mqtt_client.mqtt.client import MQTTClient
from mqtt_client.models.detection import DetectionData, DetectedObject, BoundingBox


@pytest.fixture
def mqtt_config():
    """Create a test MQTT configuration."""
    return MQTTConfig(
        broker_host="localhost",
        broker_port=1883,
        client_id="test_client",
        topic="test/topic",
        qos=0
    )


@pytest.fixture
def valid_detection_json():
    """Create a valid detection JSON payload."""
    return json.dumps({
        "detection_id": "test-123",
        "timestamp": "2023-01-01T12:00:00",
        "image_url": "http://example.com/image.jpg",
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
            }
        ]
    })


def test_mqtt_client_init(mqtt_config):
    """Test MQTT client initialization."""
    client = MQTTClient(mqtt_config)
    
    assert client.config == mqtt_config
    assert client.client is not None
    assert client._message_callback is None


@patch('paho.mqtt.client.Client')
def test_mqtt_client_connect(mock_mqtt_client, mqtt_config):
    """Test MQTT client connection."""
    # Setup mock
    mock_instance = MagicMock()
    mock_mqtt_client.return_value = mock_instance
    
    # Create client and connect
    client = MQTTClient(mqtt_config)
    client.connect()
    
    # Verify connect was called with correct parameters
    mock_instance.connect.assert_called_once_with(
        mqtt_config.broker_host,
        mqtt_config.broker_port
    )


@patch('paho.mqtt.client.Client')
def test_mqtt_client_disconnect(mock_mqtt_client, mqtt_config):
    """Test MQTT client disconnection."""
    # Setup mock
    mock_instance = MagicMock()
    mock_mqtt_client.return_value = mock_instance
    
    # Create client and disconnect
    client = MQTTClient(mqtt_config)
    client.disconnect()
    
    # Verify disconnect was called
    mock_instance.disconnect.assert_called_once()


@patch('paho.mqtt.client.Client')
def test_mqtt_client_message_callback(mock_mqtt_client, mqtt_config, valid_detection_json):
    """Test MQTT client message callback."""
    # Setup mock
    mock_instance = MagicMock()
    mock_mqtt_client.return_value = mock_instance
    
    # Create a mock message
    mock_msg = MagicMock()
    mock_msg.topic = mqtt_config.topic
    mock_msg.payload = valid_detection_json.encode('utf-8')
    
    # Create client and set message callback
    client = MQTTClient(mqtt_config)
    callback_mock = MagicMock()
    client.set_message_callback(callback_mock)
    
    # Trigger on_message callback
    client._on_message(mock_instance, None, mock_msg)
    
    # Verify callback was called with DetectionData
    callback_mock.assert_called_once()
    args, _ = callback_mock.call_args
    assert isinstance(args[0], DetectionData)
    assert args[0].detection_id == "test-123"
    assert args[0].source == "test-camera"
    assert len(args[0].objects) == 1
    assert args[0].objects[0].class_name == "person"


def test_mqtt_client_invalid_json(mqtt_config):
    """Test MQTT client with invalid JSON payload."""
    # Create client
    client = MQTTClient(mqtt_config)
    callback_mock = MagicMock()
    client.set_message_callback(callback_mock)
    
    # Create a mock message with invalid JSON
    mock_msg = MagicMock()
    mock_msg.topic = mqtt_config.topic
    mock_msg.payload = b"invalid json"
    
    # Trigger on_message callback (should not raise exception)
    client._on_message(None, None, mock_msg)
    
    # Verify callback was not called
    callback_mock.assert_not_called()
