"""Main entry point for MQTT Client application."""

import logging
import signal
import sys
import asyncio
import os
from typing import Optional, Union, Dict, List
from pathlib import Path

from mqtt_client.config import load_config
from mqtt_client.models.detection import DetectionData
from mqtt_client.models.alpr_detection import ALPREventData
from mqtt_client.models.validation import extract_license_plates, extract_image_urls
from mqtt_client.models.transformers import extract_bounding_boxes, map_alpr_to_detection_data
from mqtt_client.mqtt.client import MQTTClient
from mqtt_client.processors.image import ImageProcessor
from mqtt_client.processors.queue_manager import ImageProcessingQueue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables
mqtt_client: Optional[MQTTClient] = None
image_processor: Optional[ImageProcessor] = None
image_queue: Optional[ImageProcessingQueue] = None


def handle_detection(data: Union[DetectionData, ALPREventData]) -> None:
    """Handle validated detection data.
    
    Args:
        data: Validated detection data (either DetectionData or ALPREventData).
    """
    # Get the current event loop or create a new one if needed
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    if isinstance(data, ALPREventData):
        # Handle ALPR event data
        logger.info(f"Received ALPR event: {data.EventName}")
        logger.info(f"Camera: {data.CameraName} (ID: {data.CameraID})")
        
        # Extract license plates
        plates = extract_license_plates(data)
        logger.info(f"Detected license plates: {', '.join(plates)}")
        
        # Extract bounding boxes for all detections
        boxes = extract_bounding_boxes(data)
        logger.info(f"Found {len(boxes)} detections")
        
        # Log detection details
        for i, box in enumerate(boxes):
            logger.info(f"Detection {i+1}:")
            logger.info(f"  - License plate: {box['license_plate']}")
            logger.info(f"  - Confidence: {box['confidence']:.2f}")
            logger.info(f"  - Time: {box['timestamp']}")
            logger.info(f"  - BoundingBox: Left={box['left']}, Top={box['top']}, Right={box['right']}, Bottom={box['bottom']}")
        
        # Process images
        image_urls = extract_image_urls(data)
        for i, url in enumerate(image_urls):
            logger.info(f"Image {i+1}: {url}")
        
        # Queue detection data for image processing
        if image_queue:
            # Get the current event loop and create a task
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(image_queue.put(data))
            except Exception as e:
                logger.error(f"Error queueing ALPR data for processing: {e}")
        
    elif isinstance(data, DetectionData):
        # Handle generic detection data
        logger.info(f"Received detection data from: {data.source}")
        logger.info(f"Timestamp: {data.timestamp}")
        logger.info(f"Detected {len(data.objects)} objects")
        
        # Log detection details
        for i, obj in enumerate(data.objects):
            logger.info(f"Object {i+1}:")
            logger.info(f"  - Class: {obj.class_name} (ID: {obj.class_id})")
            logger.info(f"  - Confidence: {obj.confidence:.2f}")
            bbox = obj.bounding_box
            logger.info(f"  - BoundingBox: x={bbox.x}, y={bbox.y}, width={bbox.width}, height={bbox.height}")
        
        # Process image
        if data.image_url:
            logger.info(f"Primary image: {data.image_url}")
        
        # Process additional images
        if data.additional_image_urls:
            for i, url in enumerate(data.additional_image_urls):
                logger.info(f"Additional image {i+1}: {url}")
        
        # Queue detection data for image processing
        if image_queue:
            # Get the current event loop and create a task
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(image_queue.put(data))
            except Exception as e:
                logger.error(f"Error queueing detection data for processing: {e}")
    
    else:
        logger.warning(f"Received unknown data type: {type(data).__name__}")
        return


async def process_detection_images(data: Union[DetectionData, ALPREventData]) -> None:
    """Process images from detection data.
    
    Args:
        data: Detection data containing image URLs.
    """
    if not image_processor:
        logger.warning("Image processor not initialized, skipping image processing")
        return
    
    try:
        # Process images based on detection type
        if isinstance(data, ALPREventData):
            # Extract image URLs
            image_urls = extract_image_urls(data)
            if not image_urls:
                logger.warning("No images found in ALPR event data")
                return
            
            logger.info(f"Processing {len(image_urls)} images from ALPR event")
            
            # Get configuration for max image size
            config = load_config()
            max_size = config.image_processor.default_max_size
            
            # Process images with configured max size
            results = await image_processor.process_images(image_urls, max_size=max_size)
            
            # Log results
            successful = sum(1 for result in results.values() if result is not None)
            logger.info(f"Successfully processed {successful}/{len(image_urls)} images")
            
            # Log details for each image
            for url, result in results.items():
                if result:
                    cache_path, img = result
                    logger.info(f"Processed image: {url} -> {cache_path} ({img.width}x{img.height})")
                else:
                    logger.warning(f"Failed to process image: {url}")
            
        elif isinstance(data, DetectionData):
            # Get image URLs
            image_urls = []
            if data.image_url:
                image_urls.append(str(data.image_url))
            image_urls.extend([str(url) for url in data.additional_image_urls])
            
            if not image_urls:
                logger.warning("No images found in detection data")
                return
            
            logger.info(f"Processing {len(image_urls)} images from detection data")
            
            # Get configuration for max image size
            config = load_config()
            max_size = config.image_processor.default_max_size
            
            # Process images with configured max size
            results = await image_processor.process_images(image_urls, max_size=max_size)
            
            # Log results
            successful = sum(1 for result in results.values() if result is not None)
            logger.info(f"Successfully processed {successful}/{len(image_urls)} images")
            
            # Log details for each image
            for url, result in results.items():
                if result:
                    cache_path, img = result
                    logger.info(f"Processed image: {url} -> {cache_path} ({img.width}x{img.height})")
                else:
                    logger.warning(f"Failed to process image: {url}")
    
    except Exception as e:
        logger.error(f"Error processing images: {e}")


def handle_exit(signum, frame) -> None:
    """Handle exit signals.
    
    Args:
        signum: Signal number.
        frame: Current stack frame.
    """
    logger.info("Received exit signal, shutting down...")
    
    # Create event loop for cleanup
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Define cleanup coroutine
    async def cleanup():
        # Stop image queue
        if image_queue:
            await image_queue.stop()
        
        # Close image processor
        if image_processor:
            await image_processor.close()
        
        # Stop MQTT client
        if mqtt_client:
            mqtt_client.stop()
            mqtt_client.disconnect()
    
    # Run cleanup
    if loop.is_running():
        loop.create_task(cleanup())
    else:
        loop.run_until_complete(cleanup())
    
    sys.exit(0)


async def setup_image_processing(config) -> None:
    """Set up image processing components.
    
    Args:
        config: Application configuration.
    """
    global image_processor, image_queue
    
    # Create image processor with configuration
    image_processor = ImageProcessor(
        cache_dir=config.image_processor.cache_dir,
        max_cache_size_mb=config.image_processor.max_cache_size_mb
    )
    
    # Create image processing queue
    image_queue = ImageProcessingQueue(processor=process_detection_images)
    
    # Start image processing queue
    image_queue.start()
    
    logger.info("Image processing components initialized")


async def async_main() -> None:
    """Run the MQTT client application asynchronously."""
    global mqtt_client
    
    try:
        # Load configuration
        config = load_config()
        
        # Set up image processing
        await setup_image_processing(config)
        
        # Create and configure MQTT client
        mqtt_client = MQTTClient(config.mqtt)
        mqtt_client.set_message_callback(handle_detection)
        
        # Connect to broker and start loop
        mqtt_client.connect()
        mqtt_client.start()
        
        logger.info("MQTT client started. Press Ctrl+C to exit.")
        
        # Keep the main task alive
        while True:
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        logger.info("Main task cancelled")
    except Exception as e:
        logger.error(f"Error in async_main: {e}")
    finally:
        # Clean up resources
        if image_queue:
            await image_queue.stop()
        
        if image_processor:
            await image_processor.close()
        
        if mqtt_client:
            mqtt_client.stop()
            mqtt_client.disconnect()


def main() -> None:
    """Run the MQTT client application."""
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    try:
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run async main
        loop.run_until_complete(async_main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()
