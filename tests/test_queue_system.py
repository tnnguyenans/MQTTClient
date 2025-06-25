"""Tests for the queue system implementing producer/consumer pattern."""

import asyncio
import pytest
import os
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from mqtt_client.processors.queue_system import QueueSystem
from mqtt_client.processors.image import ImageProcessor
from mqtt_client.llm.analyzer import LLMAnalyzer
from mqtt_client.models.detection import DetectionData, DetectedObject, BoundingBox
from mqtt_client.models.alpr_detection import (
    ALPREventData, 
    DetectionRuleModel, 
    DetectionDataModel, 
    ImageInfoModel,
    ModelClassIDModel,
    ModelClassNameModel,
    DetectionModel,
    BoundingBoxModel
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "image_cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def mock_image_processor(temp_cache_dir):
    """Create a mock image processor."""
    processor = MagicMock(spec=ImageProcessor)
    
    # Mock process_images method
    async def mock_process_images(urls, max_size=None):
        results = {}
        for url in urls:
            cache_path = os.path.join(temp_cache_dir, f"image_{hash(url) % 10000}.jpg")
            results[url] = (cache_path, MagicMock())
        return results
    
    processor.process_images = mock_process_images
    processor.close = AsyncMock()
    
    return processor


@pytest.fixture
def mock_llm_analyzer():
    """Create a mock LLM analyzer."""
    analyzer = MagicMock(spec=LLMAnalyzer)
    
    # Mock analyze_detection method
    async def mock_analyze_detection(detection, image_paths):
        # Add some analysis data to the detection
        if isinstance(detection, DetectionData):
            if not detection.metadata:
                detection.metadata = {}
            detection.metadata["object_analysis"] = {
                "1": {
                    "description": "Test description",
                    "colors": ["red", "blue"],
                    "actions": ["moving"]
                }
            }
        elif isinstance(detection, ALPREventData):
            for rule in detection.Detections:
                for data in rule.DetectionData:
                    data.ExtraVisionResult = '{"description": "Test description", "colors": ["black"], "actions": ["parked"]}'
        
        return detection
    
    analyzer.analyze_detection = mock_analyze_detection
    analyzer.close = AsyncMock()
    
    return analyzer


@pytest.fixture
def mock_ui_callback():
    """Create a mock UI update callback."""
    return AsyncMock()


@pytest.fixture
def sample_detection_data():
    """Create sample detection data."""
    return DetectionData(
        source="test_source",
        timestamp="2023-01-01T12:00:00Z",
        image_url="https://example.com/image.jpg",
        objects=[
            DetectedObject(
                object_id="1",
                class_id=1,
                class_name="car",
                confidence=0.95,
                bounding_box=BoundingBox(x=100, y=100, width=200, height=150)
            )
        ]
    )


@pytest.fixture
def sample_alpr_event_data():
    """Create sample ALPR event data."""
    return ALPREventData(
        EventName="ALPR Test Event",
        CameraID=12345,
        CameraName="Test Camera",
        Detections=[
            DetectionRuleModel(
                RuleType="Presence",
                DetectionData=[
                    DetectionDataModel(
                        Name="ABC123",
                        ModelClassID=ModelClassIDModel(ModelID=1, Class=1),
                        ModelClassName=ModelClassNameModel(Model="LPR", Class=""),
                        DeploymentGroupID=1,
                        DeploymentGroupName="Test Group",
                        CameraID=12345,
                        CameraName="Test Camera",
                        CameraAddress="test.mp4",
                        TrackID=1,
                        Detections=[
                            DetectionModel(
                                Time="2023-01-01T12:00:00Z",
                                BoundingBox=BoundingBoxModel(Left=100, Top=100, Right=200, Bottom=150),
                                Score=0.95,
                                Image=1,
                                ModelClassID=ModelClassIDModel(ModelID=1, Class=1),
                                ModelClassName=ModelClassNameModel(Model="LPR", Class="ABC123"),
                                ExtraInformation="",
                                Attributes=[]
                            )
                        ],
                        Result="",
                        ExtraVisionResult="",
                        TriggeredGroup="",
                        UserInfo=None
                    )
                ],
                PipelineOption="Test Pipeline"
            )
        ],
        Images=[
            ImageInfoModel(
                ID=1,
                Image="https://example.com/image.jpg",
                ScaledFactor=1.0
            )
        ]
    )


@pytest.mark.asyncio
async def test_queue_system_initialization(
    mock_image_processor,
    mock_llm_analyzer,
    mock_ui_callback
):
    """Test queue system initialization."""
    # Create queue system
    queue_system = QueueSystem(
        image_processor=mock_image_processor,
        llm_analyzer=mock_llm_analyzer,
        ui_update_callback=mock_ui_callback
    )
    
    # Check that queues are created
    assert queue_system.mqtt_queue is not None
    assert queue_system.image_queue is not None
    assert queue_system.llm_queue is not None
    assert queue_system.ui_queue is not None
    
    # Check that processors are set
    assert queue_system.image_processor == mock_image_processor
    assert queue_system.llm_analyzer == mock_llm_analyzer
    assert queue_system.ui_update_callback == mock_ui_callback


@pytest.mark.asyncio
async def test_queue_system_start_stop(
    mock_image_processor,
    mock_llm_analyzer,
    mock_ui_callback
):
    """Test queue system start and stop."""
    # Create queue system
    queue_system = QueueSystem(
        image_processor=mock_image_processor,
        llm_analyzer=mock_llm_analyzer,
        ui_update_callback=mock_ui_callback
    )
    
    # Start queue system
    await queue_system.start()
    
    # Check that queues are running
    assert queue_system.mqtt_queue.running
    assert queue_system.image_queue.running
    assert queue_system.llm_queue.running
    assert queue_system.ui_queue.running
    
    # Stop queue system
    await queue_system.stop()
    
    # Check that queues are stopped
    assert not queue_system.mqtt_queue.running
    assert not queue_system.image_queue.running
    assert not queue_system.llm_queue.running
    assert not queue_system.ui_queue.running


@pytest.mark.asyncio
async def test_queue_system_detection_data_flow(
    mock_image_processor,
    mock_llm_analyzer,
    mock_ui_callback,
    sample_detection_data
):
    """Test queue system data flow with detection data."""
    # Create queue system
    queue_system = QueueSystem(
        image_processor=mock_image_processor,
        llm_analyzer=mock_llm_analyzer,
        ui_update_callback=mock_ui_callback
    )
    
    # Start queue system
    await queue_system.start()
    
    try:
        # Put detection data into queue
        await queue_system.put_mqtt_message(sample_detection_data)
        
        # Wait for processing to complete
        await asyncio.sleep(0.5)
        
        # Check that UI callback was called
        mock_ui_callback.assert_called()
        
        # Check that the detection data was processed through all queues
        assert queue_system.stats["mqtt_messages_received"] == 1
        assert queue_system.stats["images_processed"] > 0
        assert queue_system.stats["llm_analyses_performed"] > 0
        assert queue_system.stats["ui_updates_sent"] > 0
        
    finally:
        # Stop queue system
        await queue_system.stop()


@pytest.mark.asyncio
async def test_queue_system_alpr_event_data_flow(
    mock_image_processor,
    mock_llm_analyzer,
    mock_ui_callback,
    sample_alpr_event_data
):
    """Test queue system data flow with ALPR event data."""
    # Create queue system
    queue_system = QueueSystem(
        image_processor=mock_image_processor,
        llm_analyzer=mock_llm_analyzer,
        ui_update_callback=mock_ui_callback
    )
    
    # Start queue system
    await queue_system.start()
    
    try:
        # Put ALPR event data into queue
        await queue_system.put_mqtt_message(sample_alpr_event_data)
        
        # Wait for processing to complete
        await asyncio.sleep(0.5)
        
        # Check that UI callback was called
        mock_ui_callback.assert_called()
        
        # Check that the ALPR event data was processed through all queues
        assert queue_system.stats["mqtt_messages_received"] == 1
        assert queue_system.stats["images_processed"] > 0
        assert queue_system.stats["llm_analyses_performed"] > 0
        assert queue_system.stats["ui_updates_sent"] > 0
        
    finally:
        # Stop queue system
        await queue_system.stop()


@pytest.mark.asyncio
async def test_queue_system_without_llm(
    mock_image_processor,
    mock_ui_callback,
    sample_detection_data
):
    """Test queue system without LLM analyzer."""
    # Create queue system without LLM analyzer
    queue_system = QueueSystem(
        image_processor=mock_image_processor,
        llm_analyzer=None,
        ui_update_callback=mock_ui_callback
    )
    
    # Start queue system
    await queue_system.start()
    
    try:
        # Put detection data into queue
        await queue_system.put_mqtt_message(sample_detection_data)
        
        # Wait for processing to complete
        await asyncio.sleep(0.5)
        
        # Check that UI callback was called
        mock_ui_callback.assert_called()
        
        # Check that the detection data was processed but skipped LLM analysis
        assert queue_system.stats["mqtt_messages_received"] == 1
        assert queue_system.stats["images_processed"] > 0
        assert queue_system.stats["llm_analyses_performed"] == 0
        assert queue_system.stats["ui_updates_sent"] > 0
        
    finally:
        # Stop queue system
        await queue_system.stop()


@pytest.mark.asyncio
async def test_queue_system_error_handling(
    mock_image_processor,
    mock_ui_callback,
    sample_detection_data
):
    """Test queue system error handling."""
    # Create mock LLM analyzer that raises an exception
    mock_llm_analyzer = MagicMock(spec=LLMAnalyzer)
    
    async def mock_analyze_detection(detection, image_paths):
        raise Exception("Test exception")
    
    mock_llm_analyzer.analyze_detection = mock_analyze_detection
    mock_llm_analyzer.close = AsyncMock()
    
    # Create queue system
    queue_system = QueueSystem(
        image_processor=mock_image_processor,
        llm_analyzer=mock_llm_analyzer,
        ui_update_callback=mock_ui_callback
    )
    
    # Start queue system
    await queue_system.start()
    
    try:
        # Put detection data into queue
        await queue_system.put_mqtt_message(sample_detection_data)
        
        # Wait for processing to complete
        await asyncio.sleep(0.5)
        
        # Check that UI callback was still called despite the error
        mock_ui_callback.assert_called()
        
        # Check that an error was recorded
        assert queue_system.stats["errors"] > 0
        
    finally:
        # Stop queue system
        await queue_system.stop()


@pytest.mark.asyncio
async def test_queue_system_statistics(
    mock_image_processor,
    mock_llm_analyzer,
    mock_ui_callback,
    sample_detection_data
):
    """Test queue system statistics."""
    # Create queue system
    queue_system = QueueSystem(
        image_processor=mock_image_processor,
        llm_analyzer=mock_llm_analyzer,
        ui_update_callback=mock_ui_callback
    )
    
    # Start queue system
    await queue_system.start()
    
    try:
        # Put detection data into queue
        await queue_system.put_mqtt_message(sample_detection_data)
        
        # Wait for processing to complete
        await asyncio.sleep(0.5)
        
        # Get statistics
        stats = queue_system.get_stats()
        
        # Check statistics
        assert "mqtt_messages_received" in stats
        assert "images_processed" in stats
        assert "llm_analyses_performed" in stats
        assert "ui_updates_sent" in stats
        assert "errors" in stats
        assert "queue_sizes" in stats
        assert "uptime_seconds" in stats
        assert "avg_mqtt_time_ms" in stats
        assert "avg_image_time_ms" in stats
        assert "avg_llm_time_ms" in stats
        assert "avg_ui_time_ms" in stats
        
    finally:
        # Stop queue system
        await queue_system.stop()
