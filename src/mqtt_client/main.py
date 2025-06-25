"""Main entry point for MQTT Client application."""

import logging
import signal
import sys
import asyncio
import os
import time
from typing import Optional, Union, Dict, List, Any
from pathlib import Path

from mqtt_client.config import load_config
from mqtt_client.models.detection import DetectionData
from mqtt_client.models.alpr_detection import ALPREventData
from mqtt_client.models.validation import extract_license_plates, extract_image_urls
from mqtt_client.models.transformers import extract_bounding_boxes, map_alpr_to_detection_data
from mqtt_client.mqtt.client import MQTTClient
from mqtt_client.processors.image import ImageProcessor
from mqtt_client.processors.queue_manager import ImageProcessingQueue
from mqtt_client.llm.analyzer import LLMAnalyzer
from mqtt_client.llm.queue import LLMProcessingQueue

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
llm_analyzer: Optional[LLMAnalyzer] = None
llm_queue: Optional[LLMProcessingQueue] = None
main_event_loop: Optional[asyncio.AbstractEventLoop] = None

# Cache for processed images
processed_images_cache: Dict[str, str] = {}


def truncate_url_for_logging(url: str, max_length: int = 30) -> str:
    """Truncate long URLs or base64 data for logging purposes.
    
    Args:
        url: URL string or base64 data to truncate.
        max_length: Maximum length to display.
        
    Returns:
        str: Truncated string.
    """
    if not url:
        return ""
    
    # Handle data URI format with base64
    if url.startswith('data:') and 'base64,' in url:
        prefix, content = url.split('base64,', 1)
        return f"{prefix}base64,{content[:10]}...{content[-5:] if len(content) > 5 else ''}"
    
    # Check if it's a base64 image without prefix (check for base64 characters)
    if len(url) > 50 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in url[:20]):
        return f"{url[:10]}...{url[-5:] if len(url) > 5 else ''}"
    
    # For regular URLs, truncate if too long
    if url.startswith('http'):
        parts = url.split('/', 3)
        if len(parts) >= 4:
            return f"{parts[0]}//{parts[2]}/...{url[-10:] if len(url) > 10 else ''}"
    
    # Simple truncation for other data
    if len(url) > max_length:
        return f"{url[:max_length]}..."
        
    return url


def handle_detection(data: Union[DetectionData, ALPREventData]) -> None:
    """Handle validated detection data.
    
    Args:
        data: Validated detection data (either DetectionData or ALPREventData).
    """
    global main_event_loop
    
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
            bbox = box['bounding_box']
            logger.info(f"  - BoundingBox: x={bbox['x']}, y={bbox['y']}, width={bbox['width']}, height={bbox['height']}")
        
        # Process images
        image_urls = extract_image_urls(data)
        for i, url in enumerate(image_urls):
            logger.info(f"Image {i+1}: {truncate_url_for_logging(url)}")
            logger.debug(f"Original image URL length: {len(url)} characters")
        
        # Queue detection data for image processing
        if image_queue and main_event_loop:
            try:
                # Use thread-safe approach to schedule task on main event loop
                asyncio.run_coroutine_threadsafe(image_queue.put(data), main_event_loop)
                logger.debug("Successfully queued ALPR data for processing")
            except Exception as e:
                logger.error(f"Error queueing ALPR data for processing: {e}")
        else:
            logger.warning("Cannot queue ALPR data: image queue or main event loop not initialized")
        
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
            
        # Log image URL with truncation
        if data.image_url:
            logger.info(f"Main image: {truncate_url_for_logging(str(data.image_url))}")
            
        # Log additional image URLs with truncation
        if data.additional_image_urls:
            for i, url in enumerate(data.additional_image_urls):
                logger.info(f"Additional image {i+1}: {truncate_url_for_logging(str(url))}")
                logger.debug(f"Original additional image URL length: {len(str(url))} characters")
        
        # Process image
        if data.image_url:
            logger.info(f"Processing image: {truncate_url_for_logging(str(data.image_url))}")
            logger.debug(f"Original image URL length: {len(str(data.image_url))} characters")
        
        # Process additional images
        if data.additional_image_urls:
            for i, url in enumerate(data.additional_image_urls):
                logger.info(f"Additional image {i+1}: {truncate_url_for_logging(str(url))}")
        
        # Queue detection data for image processing
        if image_queue and main_event_loop:
            try:
                # Use thread-safe approach to schedule task on main event loop
                asyncio.run_coroutine_threadsafe(image_queue.put(data), main_event_loop)
                logger.debug("Successfully queued detection data for processing")
            except Exception as e:
                logger.error(f"Error queueing detection data for processing: {e}")
        else:
            logger.warning("Cannot queue detection data: image queue or main event loop not initialized")
    
    else:
        logger.warning(f"Received unknown data type: {type(data).__name__}")
        return


async def process_detection_images(data: Union[DetectionData, ALPREventData]) -> None:
    """Process images from detection data.
    
    Args:
        data: Detection data containing image URLs.
    """
    global processed_images_cache
    
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
            
            # Log details for each image and update cache
            image_paths = {}
            for url, result in results.items():
                if result:
                    cache_path, img = result
                    logger.info(f"Processed image: {truncate_url_for_logging(str(url))} -> {cache_path} ({img.width}x{img.height})")
                    # Update cache
                    image_paths[url] = cache_path
                    processed_images_cache[url] = cache_path
                else:
                    logger.warning(f"Failed to process image: {truncate_url_for_logging(str(url))}")
            
            # Queue for LLM analysis if we have images and LLM queue is initialized
            if image_paths and llm_queue and llm_analyzer:
                await llm_queue.put_detection(data, image_paths)
                logger.info(f"Queued ALPR event for LLM analysis")
            
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
            
            # Log details for each image and update cache
            image_paths = {}
            for url, result in results.items():
                if result:
                    cache_path, img = result
                    logger.info(f"Processed image: {truncate_url_for_logging(str(url))} -> {cache_path} ({img.width}x{img.height})")
                    # Update cache
                    image_paths[url] = cache_path
                    processed_images_cache[url] = cache_path
                else:
                    logger.warning(f"Failed to process image: {truncate_url_for_logging(str(url))}")
            
            # Queue for LLM analysis if we have images and LLM queue is initialized
            if image_paths and llm_queue and llm_analyzer:
                await llm_queue.put_detection(data, image_paths)
                logger.info(f"Queued detection data for LLM analysis")
    
    except Exception as e:
        logger.error(f"Error processing images: {e}")


def handle_exit(signum, frame) -> None:
    """Handle exit signals.
    
    Args:
        signum: Signal number.
        frame: Current stack frame.
    """
    global main_event_loop
    
    logger.info("Received exit signal, shutting down...")
    
    # Define cleanup coroutine
    async def cleanup():
        # Stop image queue
        if image_queue:
            await image_queue.stop()
        
        # Stop LLM queue
        if llm_queue:
            await llm_queue.stop()
        
        # Close image processor
        if image_processor:
            await image_processor.close()
        
        # Close LLM analyzer
        if llm_analyzer:
            await llm_analyzer.close()
        
        # Stop MQTT client
        if mqtt_client:
            mqtt_client.stop()
            mqtt_client.disconnect()
    
    # Run cleanup
    if main_event_loop and main_event_loop.is_running():
        main_event_loop.create_task(cleanup())
    else:
        # Create a new event loop for cleanup if main loop is not available
        cleanup_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(cleanup_loop)
        cleanup_loop.run_until_complete(cleanup())
    
    sys.exit(0)


async def handle_llm_analysis_result(analyzed_detection: Union[DetectionData, ALPREventData]) -> None:
    """Handle LLM analysis result.
    
    Args:
        analyzed_detection: Detection data with LLM analysis results.
    """
    try:
        if isinstance(analyzed_detection, DetectionData):
            # Log analysis results from metadata
            if analyzed_detection.metadata and "object_analysis" in analyzed_detection.metadata:
                analyses = analyzed_detection.metadata["object_analysis"]
                logger.info(f"Received LLM analysis for {len(analyses)} objects")
                
                for obj_id, analysis in analyses.items():
                    logger.info(f"Analysis for object {obj_id}:")
                    if "description" in analysis:
                        logger.info(f"  Description: {analysis['description']}")
                    if "colors" in analysis and analysis["colors"]:
                        logger.info(f"  Colors: {', '.join(analysis['colors'])}")
                    if "actions" in analysis and analysis["actions"]:
                        logger.info(f"  Actions: {', '.join(analysis['actions'])}")
        
        elif isinstance(analyzed_detection, ALPREventData):
            # Log analysis results from ExtraVisionResult
            for detection_rule in analyzed_detection.Detections:
                for detection_data in detection_rule.DetectionData:
                    if detection_data.ExtraVisionResult:
                        try:
                            import json
                            analysis = json.loads(detection_data.ExtraVisionResult)
                            
                            logger.info(f"Analysis for license plate {detection_data.Name}:")
                            if "description" in analysis:
                                logger.info(f"  Description: {analysis['description']}")
                            if "colors" in analysis and analysis["colors"]:
                                logger.info(f"  Colors: {', '.join(analysis['colors'])}")
                            if "actions" in analysis and analysis["actions"]:
                                logger.info(f"  Actions: {', '.join(analysis['actions'])}")
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in ExtraVisionResult: {detection_data.ExtraVisionResult}")
    
    except Exception as e:
        logger.error(f"Error handling LLM analysis result: {e}")


async def setup_image_processing(config) -> None:
    """Set up image processing components.
    
    Args:
        config: Application configuration.
    """
    global image_processor, image_queue, llm_analyzer, llm_queue
    
    # Create image processor with configuration
    image_processor = ImageProcessor(
        cache_dir=config.image_processor.cache_dir,
        max_cache_size_mb=config.image_processor.max_cache_size_mb
    )
    
    # Create image processing queue
    image_queue = ImageProcessingQueue(processor=process_detection_images)
    
    # Create LLM analyzer with configuration
    llm_analyzer = LLMAnalyzer(config=config.llm)
    
    # Create LLM processing queue with callback
    llm_queue = LLMProcessingQueue(
        analyzer=llm_analyzer,
        callback=handle_llm_analysis_result
    )
    
    # Start image processing queue
    await image_queue.start()
    
    # Start LLM processing queue
    await llm_queue.start()
    
    logger.info("Image processing and LLM components initialized")


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
    global main_event_loop
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    try:
        # Create a new event loop and store it globally
        main_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(main_event_loop)
        
        # Run async main
        main_event_loop.run_until_complete(async_main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()
