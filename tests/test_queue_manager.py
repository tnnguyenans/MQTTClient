"""Unit tests for the queue manager module."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import time

from mqtt_client.processors.queue_manager import ProcessingQueue, ImageProcessingQueue
from mqtt_client.models.alpr_detection import ALPREventData, ImageInfoModel
from mqtt_client.models.detection import DetectionData


@pytest.fixture
def mock_processor():
    """Create a mock processor function for testing."""
    return AsyncMock()


@pytest.fixture
def processing_queue(mock_processor):
    """Create a processing queue instance for testing."""
    queue = ProcessingQueue(name="test_queue", max_size=10, processor=mock_processor)
    yield queue
    
    # Cleanup
    if queue._task and not queue._task.done():
        asyncio.run(queue.stop())


@pytest.fixture
def image_processing_queue(mock_processor):
    """Create an image processing queue instance for testing."""
    queue = ImageProcessingQueue(processor=mock_processor, max_size=10)
    yield queue
    
    # Cleanup
    if queue._task and not queue._task.done():
        asyncio.run(queue.stop())


@pytest.fixture
def mock_alpr_data():
    """Create mock ALPR event data for testing."""
    return ALPREventData(
        EventName="Test ALPR Event",
        CameraID=12345,
        CameraName="Test Camera",
        Detections=[],  # Empty for this test
        Images=[
            ImageInfoModel(ID=1, Image="http://example.com/image1.jpg", ScaledFactor=1.0),
            ImageInfoModel(ID=2, Image="http://example.com/image2.jpg", ScaledFactor=1.0),
        ]
    )


@pytest.fixture
def mock_detection_data():
    """Create mock detection data for testing."""
    return DetectionData(
        source="test_source",
        timestamp="2025-06-25T12:00:00Z",
        objects=[],
        image_url="http://example.com/image.jpg",
        additional_image_urls=["http://example.com/additional1.jpg"]
    )


class TestProcessingQueue:
    """Test cases for the ProcessingQueue class."""

    @pytest.mark.asyncio
    async def test_queue_initialization(self, processing_queue):
        """Test queue initialization."""
        assert processing_queue._name == "test_queue"
        assert processing_queue._max_size == 10
        assert processing_queue._queue is not None
        assert processing_queue._task is None
        assert processing_queue._running is False

    @pytest.mark.asyncio
    async def test_start_stop(self, processing_queue):
        """Test starting and stopping the queue."""
        # Start the queue
        processing_queue.start()
        assert processing_queue._running is True
        assert processing_queue._task is not None
        
        # Stop the queue
        await processing_queue.stop()
        assert processing_queue._running is False
        assert processing_queue._task.done()

    @pytest.mark.asyncio
    async def test_put_and_process(self, processing_queue, mock_processor):
        """Test putting items in the queue and processing them."""
        # Start the queue
        processing_queue.start()
        
        # Put an item in the queue
        test_item = {"test": "data"}
        await processing_queue.put(test_item)
        
        # Wait a bit for processing
        await asyncio.sleep(0.1)
        
        # Verify the processor was called with the item
        mock_processor.assert_called_once_with(test_item)
        
        # Stop the queue
        await processing_queue.stop()

    @pytest.mark.asyncio
    async def test_queue_size(self, processing_queue):
        """Test queue size reporting."""
        # Start with empty queue
        assert processing_queue.size() == 0
        assert processing_queue.is_empty() is True
        
        # Add items without starting the queue (so they won't be processed)
        await processing_queue._queue.put("item1")
        await processing_queue._queue.put("item2")
        
        # Check size
        assert processing_queue.size() == 2
        assert processing_queue.is_empty() is False
        
        # Clear the queue for cleanup
        while not processing_queue._queue.empty():
            await processing_queue._queue.get()

    @pytest.mark.asyncio
    async def test_processor_exception_handling(self, mock_processor):
        """Test that exceptions in the processor are handled properly."""
        # Configure the mock processor to raise an exception
        mock_processor.side_effect = Exception("Test exception")
        
        # Create a queue with the failing processor
        queue = ProcessingQueue(name="test_exception", max_size=10, processor=mock_processor)
        
        try:
            # Start the queue
            queue.start()
            
            # Put an item in the queue
            await queue.put("test_item")
            
            # Wait a bit for processing
            await asyncio.sleep(0.1)
            
            # Verify the processor was called
            mock_processor.assert_called_once()
            
            # Queue should still be running despite the exception
            assert queue._running is True
        finally:
            # Stop the queue
            await queue.stop()


class TestImageProcessingQueue:
    """Test cases for the ImageProcessingQueue class."""

    @pytest.mark.asyncio
    async def test_image_queue_initialization(self, image_processing_queue):
        """Test image queue initialization."""
        assert image_processing_queue._name == "image_processing"
        assert image_processing_queue._max_size == 10
        assert image_processing_queue._queue is not None
        assert image_processing_queue._task is None
        assert image_processing_queue._running is False

    @pytest.mark.asyncio
    async def test_process_alpr_data(self, image_processing_queue, mock_processor, mock_alpr_data):
        """Test processing ALPR event data."""
        # Start the queue
        image_processing_queue.start()
        
        # Put ALPR data in the queue
        await image_processing_queue.put(mock_alpr_data)
        
        # Wait a bit for processing
        await asyncio.sleep(0.1)
        
        # Verify the processor was called with the ALPR data
        mock_processor.assert_called_once_with(mock_alpr_data)
        
        # Stop the queue
        await image_processing_queue.stop()

    @pytest.mark.asyncio
    async def test_process_detection_data(self, image_processing_queue, mock_processor, mock_detection_data):
        """Test processing generic detection data."""
        # Start the queue
        image_processing_queue.start()
        
        # Put detection data in the queue
        await image_processing_queue.put(mock_detection_data)
        
        # Wait a bit for processing
        await asyncio.sleep(0.1)
        
        # Verify the processor was called with the detection data
        mock_processor.assert_called_once_with(mock_detection_data)
        
        # Stop the queue
        await image_processing_queue.stop()

    @pytest.mark.asyncio
    async def test_queue_backpressure(self, mock_processor):
        """Test queue backpressure when max size is reached."""
        # Create a queue with small max size
        queue = ImageProcessingQueue(processor=mock_processor, max_size=2)
        
        # Configure the processor to be slow
        async def slow_processor(item):
            await asyncio.sleep(0.5)
            return item
        
        queue._processor = slow_processor
        
        try:
            # Start the queue
            queue.start()
            
            # Put items in the queue quickly to fill it up
            await queue.put("item1")
            await queue.put("item2")
            
            # This should block due to backpressure
            # Use wait_for with a timeout to test this
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(queue.put("item3"), timeout=0.1)
        finally:
            # Stop the queue
            await queue.stop()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
