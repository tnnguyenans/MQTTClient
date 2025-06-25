"""Test script for the image processing queue feature."""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root directory to path to allow imports
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mqtt_client.config import load_config
from mqtt_client.processors.image import ImageProcessor
from mqtt_client.processors.queue_manager import ImageProcessingQueue
from mqtt_client.models.alpr_detection import ALPREventData, ImageInfoModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Sample image URLs for testing
TEST_IMAGES = [
    "https://images.unsplash.com/photo-1578983427937-26078ee3d9d3?q=80&w=1000",  # Car with license plate
    "https://images.unsplash.com/photo-1580273916550-e323be2ae537?q=80&w=1000",  # Car on road
]


async def process_detection_data(data):
    """Process detection data (simulating the function in main.py)."""
    logger.info(f"Processing detection data: {data.EventName}")
    
    # Load configuration
    config = load_config()
    
    # Create image processor
    image_processor = ImageProcessor(
        cache_dir=config.image_processor.cache_dir,
        max_cache_size_mb=config.image_processor.max_cache_size_mb
    )
    
    try:
        # Extract image URLs
        image_urls = [str(image.Image) for image in data.Images]
        logger.info(f"Processing {len(image_urls)} images from event")
        
        # Process images
        results = await image_processor.process_images(
            image_urls, 
            max_size=config.image_processor.default_max_size
        )
        
        # Log results
        successful = sum(1 for result in results.values() if result is not None)
        logger.info(f"Successfully processed {successful}/{len(image_urls)} images")
        
        # Simulate some processing time
        await asyncio.sleep(1)
        
        return results
    
    finally:
        await image_processor.close()


async def test_queue_manager():
    """Test the queue manager functionality."""
    # Create image processing queue
    image_queue = ImageProcessingQueue(processor=process_detection_data)
    
    try:
        # Start the queue
        image_queue.start()
        logger.info("Image processing queue started")
        
        # Create mock ALPR event data items
        mock_events = []
        for i in range(5):
            mock_event = ALPREventData(
                EventName=f"Test ALPR Event {i+1}",
                CameraID=12345,
                CameraName="Test Camera",
                Detections=[],  # Empty for this test
                Images=[
                    ImageInfoModel(ID=1, Image=TEST_IMAGES[0], ScaledFactor=1.0),
                    ImageInfoModel(ID=2, Image=TEST_IMAGES[1], ScaledFactor=1.0),
                ]
            )
            mock_events.append(mock_event)
        
        # Add events to the queue
        logger.info("Adding events to the queue...")
        for event in mock_events:
            await image_queue.put(event)
            logger.info(f"Added event to queue: {event.EventName}")
        
        # Wait for all events to be processed
        logger.info("Waiting for queue to process all events...")
        while not image_queue.is_empty():
            logger.info(f"Queue size: {image_queue.size()}")
            await asyncio.sleep(1)
        
        # Wait a bit more to ensure all processing is complete
        await asyncio.sleep(3)
        logger.info("All events processed")
    
    finally:
        # Stop the queue
        await image_queue.stop()
        logger.info("Image processing queue stopped")


async def test_main():
    """Main test function."""
    try:
        await test_queue_manager()
    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True)


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_main())
