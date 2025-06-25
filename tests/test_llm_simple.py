"""Simplified tests for LLM analyzer integration."""

import os
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any

from mqtt_client.config import AppConfig, LLMConfig, MQTTConfig, ImageProcessorConfig
from mqtt_client.models.detection import DetectionData, DetectedObject, BoundingBox
from mqtt_client.llm.analyzer import LLMAnalyzer
from mqtt_client.llm.queue import LLMProcessingQueue


@pytest.fixture
def app_config():
    """Create a test application configuration."""
    return AppConfig(
        llm=LLMConfig(
            provider="openai",
            api_key="test_key",
            model="gpt-4-vision-preview",
            ollama_base_url="http://localhost:11434",
            rate_limit=10,
            max_retries=3
        ),
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
        image_url="https://example.com/image.jpg",
        additional_image_urls=[],
        metadata={}  # Initialize metadata as empty dict
    )


@pytest.fixture
def image_paths():
    """Create test image paths mapping."""
    return {
        "https://example.com/image.jpg": "/tmp/cache/image1.jpg"
    }


class TestLLMAnalyzer:
    """Test LLM analyzer functionality."""
    
    @pytest.mark.asyncio
    async def test_analyze_detection(self, app_config, detection_data, image_paths):
        """Test analyzing detection data with mocked analyze_object."""
        # Create analyzer
        analyzer = LLMAnalyzer(config=app_config.llm)
        
        # Mock the analyze_object method
        async def mock_analyze_object(image_path, object_info):
            """Mock implementation that returns analysis results."""
            class_name = object_info.get("class_name", "object")
            
            # Create different results based on object type
            if class_name == "person":
                analysis = {
                    "description": "A person walking",
                    "colors": ["blue", "black"],
                    "actions": ["walking"],
                    "attributes": []
                }
            else:
                analysis = {
                    "description": "A red car parked",
                    "colors": ["red"],
                    "actions": ["parked"],
                    "attributes": []
                }
            
            # Return object info with analysis
            return {
                **object_info,
                "llm_analysis": analysis
            }
        
        # Apply the mock
        with patch.object(analyzer, "analyze_object", side_effect=mock_analyze_object):
            # Analyze detection
            result = await analyzer.analyze_detection(detection_data, image_paths)
            
            # Check that metadata was updated
            assert "object_analysis" in result.metadata
            assert "obj1" in result.metadata["object_analysis"]
            assert "obj2" in result.metadata["object_analysis"]
            
            # Check analysis content for person
            obj1 = result.metadata["object_analysis"]["obj1"]
            assert obj1["description"] == "A person walking"
            assert "blue" in obj1["colors"]
            assert "walking" in obj1["actions"]
            
            # Check analysis content for car
            obj2 = result.metadata["object_analysis"]["obj2"]
            assert obj2["description"] == "A red car parked"
            assert "red" in obj2["colors"]
            assert "parked" in obj2["actions"]


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
    # Create analyzer
    analyzer = LLMAnalyzer(config=app_config.llm)
    
    # Mock the analyze_object method
    async def mock_analyze_object(image_path, object_info):
        """Mock implementation that returns analysis results."""
        class_name = object_info.get("class_name", "object")
        
        # Create analysis results
        analysis = {
            "description": f"A {class_name} detected",
            "colors": ["blue"],
            "actions": ["standing"],
            "attributes": []
        }
        
        # Return object info with analysis
        return {
            **object_info,
            "llm_analysis": analysis
        }
    
    # Apply the mock
    with patch.object(analyzer, "analyze_object", side_effect=mock_analyze_object):
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
            assert "description" in obj1
            assert "colors" in obj1
            assert "actions" in obj1
        finally:
            # Stop queue
            await queue.stop()
