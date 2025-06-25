"""Tests for LLM analyzer integration."""

import os
import json
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
from typing import Dict, Any, List

import aiohttp
from pydantic import HttpUrl

from mqtt_client.config import AppConfig, LLMConfig, MQTTConfig, ImageProcessorConfig
from mqtt_client.models.detection import DetectionData, DetectedObject, BoundingBox
from mqtt_client.models.alpr_detection import ALPREventData
from mqtt_client.llm.analyzer import LLMAnalyzer
from mqtt_client.llm.queue import LLMProcessingQueue


@pytest.fixture
def llm_config():
    """Create a test LLM configuration."""
    return LLMConfig(
        provider="openai",
        api_key="test_key",
        model="gpt-4o",
        ollama_base_url="http://localhost:11434",
        rate_limit=10,
        max_retries=3
    )


@pytest.fixture
def app_config(llm_config):
    """Create a test application configuration."""
    return AppConfig(
        llm=llm_config,
        mqtt=MQTTConfig(
            broker_host="localhost",
            broker_port=1883,
            client_id="test_client",
            topic="test/topic",
            qos=0
        ),
        image_processor=ImageProcessorConfig(
            cache_dir="/tmp/test_cache",
            max_cache_size_mb=100,
            default_max_size=(800, 600)
        )
    )


@pytest.fixture
def detection_data():
    """Create test detection data."""
    return DetectionData(
        source="test_camera",
        objects=[
            DetectedObject(
                class_id=1,
                class_name="person",
                confidence=0.95,
                bounding_box=BoundingBox(
                    x=100,
                    y=100,
                    width=200,
                    height=300
                ),
                object_id="obj1"
            ),
            DetectedObject(
                class_id=2,
                class_name="car",
                confidence=0.85,
                bounding_box=BoundingBox(
                    x=400,
                    y=300,
                    width=250,
                    height=180
                ),
                object_id="obj2"
            )
        ],
        image_url=HttpUrl("https://example.com/image.jpg"),
        additional_image_urls=[]
    )


@pytest.fixture
def alpr_event_data():
    """Create test ALPR event data."""
    # Load sample ALPR data from file
    test_dir = Path(__file__).parent
    sample_file = test_dir / "data" / "sample_alpr_event.json"
    
    # If sample file doesn't exist, create a minimal ALPR event
    if not sample_file.exists():
        return ALPREventData(
            EventName="Test ALPR Event",
            CameraID=12345,
            CameraName="Test Camera",
            Detections=[],
            Images=[]
        )
    
    # Load from file
    with open(sample_file, "r") as f:
        data = json.load(f)
    
    return ALPREventData(**data)


@pytest.fixture
def image_paths():
    """Create test image paths mapping."""
    return {
        "https://example.com/image.jpg": "/tmp/cache/image1.jpg",
        "https://example.com/image2.jpg": "/tmp/cache/image2.jpg"
    }


class TestLLMAnalyzer:
    """Test LLM analyzer functionality."""
    
    @pytest.mark.asyncio
    async def test_analyze_detection_with_openai(self, app_config, detection_data, image_paths):
        """Test analyzing detection data with OpenAI."""
        # Mock OpenAI response
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "objects": {
                                "obj1": {
                                    "description": "A person walking",
                                    "colors": ["blue", "black"],
                                    "actions": ["walking"]
                                },
                                "obj2": {
                                    "description": "A red car parked",
                                    "colors": ["red"],
                                    "actions": ["parked"]
                                }
                            }
                        })
                    }
                }
            ]
        }
        
        # Create analyzer with mocked session
        with patch("aiohttp.ClientSession") as mock_session:
            # Set up the mock
            mock_session_instance = AsyncMock()
            mock_session.return_value = mock_session_instance
            
            # Mock the post method
            mock_response_obj = AsyncMock()
            mock_response_obj.status = 200
            mock_response_obj.json.return_value = mock_response
            
            mock_session_instance.post.return_value.__aenter__.return_value = mock_response_obj
            
            # Create analyzer
            analyzer = LLMAnalyzer(config=app_config.llm)
            
            # Mock image cropping
            with patch.object(analyzer, "_crop_object_from_image", return_value=("/tmp/cropped.jpg", MagicMock())):
                # Analyze detection
                result = await analyzer.analyze_detection(detection_data, image_paths)
                
                # Check that OpenAI API was called
                mock_session_instance.post.assert_called_once()
                
                # Check that metadata was updated
                assert "object_analysis" in result.metadata
                assert "obj1" in result.metadata["object_analysis"]
                assert "obj2" in result.metadata["object_analysis"]
                
                # Check analysis content
                obj1 = result.metadata["object_analysis"]["obj1"]
                assert obj1["description"] == "A person walking"
                assert "blue" in obj1["colors"]
                assert "walking" in obj1["actions"]
                
                obj2 = result.metadata["object_analysis"]["obj2"]
                assert obj2["description"] == "A red car parked"
                assert "red" in obj2["colors"]
                assert "parked" in obj2["actions"]
    
    @pytest.mark.asyncio
    async def test_analyze_detection_with_ollama(self, app_config, detection_data, image_paths):
        """Test analyzing detection data with Ollama."""
        # Set provider to ollama
        app_config.llm.provider = "ollama"
        
        # Mock Ollama response
        mock_response = {
            "response": json.dumps({
                "objects": {
                    "obj1": {
                        "description": "A person standing",
                        "colors": ["green", "white"],
                        "actions": ["standing"]
                    },
                    "obj2": {
                        "description": "A blue car moving",
                        "colors": ["blue"],
                        "actions": ["moving"]
                    }
                }
            })
        }
        
        # Create analyzer with mocked session
        with patch("aiohttp.ClientSession") as mock_session:
            # Set up the mock
            mock_session_instance = AsyncMock()
            mock_session.return_value = mock_session_instance
            
            # Mock the post method
            mock_response_obj = AsyncMock()
            mock_response_obj.status = 200
            mock_response_obj.json.return_value = mock_response
            
            mock_session_instance.post.return_value.__aenter__.return_value = mock_response_obj
            
            # Create analyzer
            analyzer = LLMAnalyzer(config=app_config.llm)
            
            # Mock image cropping
            with patch.object(analyzer, "_crop_object_from_image", return_value=("/tmp/cropped.jpg", MagicMock())):
                # Analyze detection
                result = await analyzer.analyze_detection(detection_data, image_paths)
                
                # Check that Ollama API was called
                mock_session_instance.post.assert_called_once()
                
                # Check that metadata was updated
                assert "object_analysis" in result.metadata
                assert "obj1" in result.metadata["object_analysis"]
                assert "obj2" in result.metadata["object_analysis"]
                
                # Check analysis content
                obj1 = result.metadata["object_analysis"]["obj1"]
                assert obj1["description"] == "A person standing"
                assert "green" in obj1["colors"]
                assert "standing" in obj1["actions"]
                
                obj2 = result.metadata["object_analysis"]["obj2"]
                assert obj2["description"] == "A blue car moving"
                assert "blue" in obj2["colors"]
                assert "moving" in obj2["actions"]


class TestLLMProcessingQueue:
    """Test LLM processing queue functionality."""
    
    @pytest.mark.asyncio
    async def test_queue_processing(self, app_config, detection_data, image_paths):
        """Test that queue processes items correctly."""
        # Create mock analyzer
        mock_analyzer = AsyncMock()
        mock_analyzer.analyze_detection.return_value = detection_data
        
        # Create mock callback
        mock_callback = AsyncMock()
        
        # Create queue
        queue = LLMProcessingQueue(
            analyzer=mock_analyzer,
            callback=mock_callback
        )
        
        try:
            # Start queue
            await queue.start()
            
            # Add item to queue
            await queue.put_detection(detection_data, image_paths)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Check that analyzer was called
            mock_analyzer.analyze_detection.assert_called_once_with(detection_data, image_paths)
            
            # Check that callback was called
            mock_callback.assert_called_once()
        finally:
            # Stop queue
            await queue.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, app_config, detection_data, image_paths):
        """Test that queue handles errors gracefully."""
        # Create mock analyzer that raises an exception
        mock_analyzer = AsyncMock()
        mock_analyzer.analyze_detection.side_effect = Exception("Test error")
        
        # Create mock callback
        mock_callback = AsyncMock()
        
        # Create queue
        queue = LLMProcessingQueue(
            analyzer=mock_analyzer,
            callback=mock_callback
        )
        
        try:
            # Start queue
            await queue.start()
            
            # Add item to queue
            await queue.put_detection(detection_data, image_paths)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Check that analyzer was called
            mock_analyzer.analyze_detection.assert_called_once_with(detection_data, image_paths)
            
            # Check that callback was not called due to error
            mock_callback.assert_not_called()
        finally:
            # Stop queue
            await queue.stop()


@pytest.mark.asyncio
async def test_end_to_end_integration(app_config, detection_data, image_paths):
    """Test end-to-end integration of LLM analyzer and queue."""
    # Mock OpenAI response
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "objects": {
                            "obj1": {
                                "description": "A person walking",
                                "colors": ["blue", "black"],
                                "actions": ["walking"]
                            }
                        }
                    })
                }
            }
        ]
    }
    
    # Create analyzer with mocked session
    with patch("aiohttp.ClientSession") as mock_session:
        # Set up the mock
        mock_session_instance = AsyncMock()
        mock_session.return_value = mock_session_instance
        
        # Mock the post method
        mock_response_obj = AsyncMock()
        mock_response_obj.status = 200
        mock_response_obj.json.return_value = mock_response
        
        mock_session_instance.post.return_value.__aenter__.return_value = mock_response_obj
        
        # Create analyzer
        analyzer = LLMAnalyzer(config=app_config.llm)
        
        # Mock image cropping
        with patch.object(analyzer, "_crop_object_from_image", return_value=("/tmp/cropped.jpg", MagicMock())):
            # Create result collector
            results = []
            
            async def collect_result(result):
                results.append(result)
            
            # Create queue
            queue = LLMProcessingQueue(
                analyzer=analyzer,
                callback=collect_result
            )
            
            try:
                # Start queue
                await queue.start()
                
                # Add item to queue
                await queue.put_detection(detection_data, image_paths)
                
                # Wait for processing
                await asyncio.sleep(0.5)
                
                # Check that result was collected
                assert len(results) == 1
                
                # Check that metadata was updated
                result = results[0]
                assert "object_analysis" in result.metadata
                assert "obj1" in result.metadata["object_analysis"]
                
                # Check analysis content
                obj1 = result.metadata["object_analysis"]["obj1"]
                assert obj1["description"] == "A person walking"
                assert "blue" in obj1["colors"]
                assert "walking" in obj1["actions"]
            finally:
                # Stop queue
                await queue.stop()