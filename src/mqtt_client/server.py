"""FastAPI server for MQTT Client application."""

import asyncio
import logging
import os
import signal
import sys
import uvicorn
from typing import Dict, Any, Optional, Union

from mqtt_client.config import load_config
from mqtt_client.models.detection import DetectionData
from mqtt_client.models.alpr_detection import ALPREventData
from mqtt_client.mqtt.client import MQTTClient
from mqtt_client.processors.image import ImageProcessor
from mqtt_client.processors.queue_manager import ImageProcessingQueue
from mqtt_client.llm.analyzer import LLMAnalyzer
from mqtt_client.llm.queue import LLMProcessingQueue
from mqtt_client.api.app import create_app
from mqtt_client.api.routes import handle_detection_update

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


async def handle_llm_analysis_result(analyzed_detection: Union[DetectionData, ALPREventData]) -> None:
    """Handle LLM analysis result.
    
    Args:
        analyzed_detection: Detection data with LLM analysis results.
    """
    try:
        # Log the detection results (simplified from main.py)
        if isinstance(analyzed_detection, DetectionData):
            logger.info(f"Received LLM analysis for DetectionData from {analyzed_detection.source}")
        elif isinstance(analyzed_detection, ALPREventData):
            logger.info(f"Received LLM analysis for ALPREventData from {analyzed_detection.CameraName}")
        
        # Forward to API for WebSocket broadcasting
        await handle_detection_update(analyzed_detection)
        
    except Exception as e:
        logger.error(f"Error handling LLM analysis result: {e}")


async def process_detection_images(data: Union[DetectionData, ALPREventData]) -> None:
    """Process images from detection data.
    
    Args:
        data: Detection data containing image URLs.
    """
    if not image_processor:
        logger.warning("Image processor not initialized, skipping image processing")
        return
    
    try:
        # Process images based on detection type (simplified from main.py)
        if isinstance(data, ALPREventData):
            # Extract and process images
            from mqtt_client.models.validation import extract_image_urls
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
            
            # Queue for LLM analysis if we have images and LLM queue is initialized
            image_paths = {url: path_img[0] for url, path_img in results.items() if path_img}
            
            if image_paths and llm_queue and llm_analyzer:
                await llm_queue.put_detection(data, image_paths)
                logger.info(f"Queued ALPR event for LLM analysis")
        
        elif isinstance(data, DetectionData):
            # Process the main image and additional images
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
            
            # Queue for LLM analysis if we have images and LLM queue is initialized
            image_paths = {url: path_img[0] for url, path_img in results.items() if path_img}
            
            if image_paths and llm_queue and llm_analyzer:
                await llm_queue.put_detection(data, image_paths)
                logger.info(f"Queued detection data for LLM analysis")
    
    except Exception as e:
        logger.error(f"Error processing images: {e}")


def handle_detection(data: Union[DetectionData, ALPREventData]) -> None:
    """Handle validated detection data from MQTT.
    
    Args:
        data: Validated detection data (either DetectionData or ALPREventData).
    """
    global main_event_loop
    
    if isinstance(data, ALPREventData):
        # Log received ALPR event (simplified)
        logger.info(f"Received ALPR event from {data.CameraName}")
    elif isinstance(data, DetectionData):
        # Log received detection data (simplified)
        logger.info(f"Received detection data with {len(data.objects)} objects")
    
    # Queue detection data for image processing
    if image_queue and main_event_loop:
        try:
            # Use thread-safe approach to schedule task on main event loop
            asyncio.run_coroutine_threadsafe(image_queue.put(data), main_event_loop)
            logger.debug("Successfully queued data for processing")
        except Exception as e:
            logger.error(f"Error queueing data for processing: {e}")
    else:
        logger.warning("Cannot queue data: image queue or main event loop not initialized")


async def setup_components(config) -> None:
    """Set up all application components.
    
    Args:
        config: Application configuration.
    """
    global image_processor, image_queue, llm_analyzer, llm_queue, mqtt_client
    
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
    
    # Create and configure MQTT client
    mqtt_client = MQTTClient(config.mqtt)
    mqtt_client.set_message_callback(handle_detection)
    
    # Connect to broker and start loop
    connection_success = mqtt_client.connect()
    if connection_success:
        mqtt_client.start()
    else:
        logger.warning("MQTT client failed to connect, continuing without MQTT functionality")
        # We'll keep the mqtt_client reference but it won't be actively connected
    
    logger.info("All components initialized")


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
            try:
                mqtt_client.stop()
                mqtt_client.disconnect()
            except Exception as e:
                logger.warning(f"Error during MQTT client cleanup: {e}")
                # Continue with cleanup even if MQTT client fails
    
    # Run cleanup
    if main_event_loop and main_event_loop.is_running():
        main_event_loop.create_task(cleanup())
    else:
        # Create a new event loop for cleanup if main loop is not available
        cleanup_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(cleanup_loop)
        cleanup_loop.run_until_complete(cleanup())
    
    sys.exit(0)


async def setup_server() -> None:
    """Set up server and all components."""
    global main_event_loop
    
    # Load configuration
    config = load_config()
    
    # Set up components
    await setup_components(config)
    
    logger.info("Server setup completed")


def main() -> None:
    """Run the server application with FastAPI and MQTT client."""
    global main_event_loop
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    try:
        # Load configuration
        config = load_config()
        
        # Create a new event loop and store it globally
        main_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(main_event_loop)
        
        # Set up server components
        main_event_loop.run_until_complete(setup_server())
        
        # Create FastAPI app
        app = create_app()
        
        # Configure uvicorn
        # Explicitly set port to 8088 to override any configuration
        port = 8088
        host = config.api.host if hasattr(config, 'api') and hasattr(config.api, 'host') else "0.0.0.0"
        
        logger.info(f"Starting server on {host}:{port}")
        
        # Use a different approach that works with our existing event loop
        from uvicorn.config import Config
        from uvicorn.server import Server
        
        config = Config(app=app, host=host, port=port, log_level="info")
        server = Server(config)
        
        # Run the server in the same event loop
        main_event_loop.run_until_complete(server.serve())
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()