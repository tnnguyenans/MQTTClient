"""Queue system implementing producer/consumer pattern for MQTT client application.

This module implements a comprehensive producer/consumer pattern using asyncio queues
to handle MQTT message ingestion, image processing, LLM analysis, and UI updates.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, Awaitable, TypeVar, Generic, Union, List, Type

from mqtt_client.models.detection import DetectionData
from mqtt_client.models.alpr_detection import ALPREventData
from mqtt_client.processors.queue_manager import ProcessingQueue
from mqtt_client.processors.image import ImageProcessor
from mqtt_client.llm.analyzer import LLMAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for queue items
T = TypeVar('T')


class QueueSystem:
    """Main queue system implementing producer/consumer pattern."""
    
    def __init__(
        self,
        image_processor: ImageProcessor,
        llm_analyzer: Optional[LLMAnalyzer] = None,
        ui_update_callback: Optional[Callable[[Union[DetectionData, ALPREventData]], Awaitable[None]]] = None,
        max_queue_size: int = 100
    ):
        """Initialize queue system.
        
        Args:
            image_processor: Image processor instance.
            llm_analyzer: Optional LLM analyzer instance.
            ui_update_callback: Optional callback for UI updates.
            max_queue_size: Maximum size for all queues.
        """
        self.image_processor = image_processor
        self.llm_analyzer = llm_analyzer
        self.ui_update_callback = ui_update_callback
        self.max_queue_size = max_queue_size
        
        # Create queues
        self.mqtt_queue = ProcessingQueue[Union[DetectionData, ALPREventData]](
            name="mqtt_producer",
            max_size=max_queue_size,
            processor=self._process_mqtt_message
        )
        
        self.image_queue = ProcessingQueue[Dict[str, Any]](
            name="image_consumer",
            max_size=max_queue_size,
            processor=self._process_image
        )
        
        self.llm_queue = ProcessingQueue[Dict[str, Any]](
            name="llm_consumer",
            max_size=max_queue_size,
            processor=self._process_llm_analysis
        )
        
        self.ui_queue = ProcessingQueue[Union[DetectionData, ALPREventData]](
            name="ui_consumer",
            max_size=max_queue_size,
            processor=self._process_ui_update
        )
        
        # Cache for processed images
        self.processed_images_cache: Dict[str, str] = {}
        
        # Statistics
        self.stats = {
            "mqtt_messages_received": 0,
            "images_processed": 0,
            "llm_analyses_performed": 0,
            "ui_updates_sent": 0,
            "errors": 0,
            "start_time": None,
            "processing_times": {
                "mqtt": [],
                "image": [],
                "llm": [],
                "ui": []
            }
        }
        
        logger.info("Queue system initialized")
    
    async def start(self) -> None:
        """Start all queue processors."""
        self.stats["start_time"] = time.time()
        
        # Start queues in reverse order (consumers first)
        await self.ui_queue.start()
        
        if self.llm_analyzer:
            await self.llm_queue.start()
        
        await self.image_queue.start()
        await self.mqtt_queue.start()
        
        logger.info("Queue system started")
    
    async def stop(self) -> None:
        """Stop all queue processors."""
        # Stop queues in order (producer first)
        await self.mqtt_queue.stop()
        await self.image_queue.stop()
        
        if self.llm_analyzer:
            await self.llm_queue.stop()
        
        await self.ui_queue.stop()
        
        logger.info("Queue system stopped")
    
    async def put_mqtt_message(self, data: Union[DetectionData, ALPREventData]) -> None:
        """Add MQTT message to queue.
        
        Args:
            data: Validated detection data.
        """
        self.stats["mqtt_messages_received"] += 1
        await self.mqtt_queue.put(data)
    
    async def _process_mqtt_message(self, data: Union[DetectionData, ALPREventData]) -> None:
        """Process MQTT message.
        
        Args:
            data: Validated detection data.
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Processing MQTT message: {type(data).__name__}")
            
            # Create item for image processing queue
            item = {
                "detection": data,
                "timestamp": time.time()
            }
            
            # Add to image processing queue
            await self.image_queue.put(item)
            
            # Update stats
            processing_time = (time.time() - start_time) * 1000  # ms
            self.stats["processing_times"]["mqtt"].append(processing_time)
            logger.debug(f"MQTT message queued for image processing in {processing_time:.2f} ms")
            
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
            self.stats["errors"] += 1
    
    async def _process_image(self, item: Dict[str, Any]) -> None:
        """Process images from detection data.
        
        Args:
            item: Queue item containing detection data.
        """
        start_time = time.time()
        
        try:
            detection = item["detection"]
            logger.debug(f"Processing images for: {type(detection).__name__}")
            
            # Extract image URLs based on detection type
            image_urls = []
            
            if isinstance(detection, ALPREventData):
                # Extract image URLs from ALPR event data
                for image in detection.Images:
                    if image.Image:
                        image_urls.append(str(image.Image))
            
            elif isinstance(detection, DetectionData):
                # Extract image URL from detection data
                if detection.image_url:
                    image_urls.append(str(detection.image_url))
                
                # Add additional image URLs
                image_urls.extend([str(url) for url in detection.additional_image_urls])
            
            if not image_urls:
                logger.warning("No images found in detection data")
                return
            
            logger.debug(f"Processing {len(image_urls)} images")
            
            # Process images
            results = await self.image_processor.process_images(image_urls)
            
            # Update cache and create image paths dictionary
            image_paths = {}
            for url, result in results.items():
                if result:
                    cache_path, _ = result
                    image_paths[url] = cache_path
                    self.processed_images_cache[url] = cache_path
            
            # Update stats
            self.stats["images_processed"] += len(image_paths)
            
            # Skip LLM analysis if no images were processed
            if not image_paths:
                logger.warning("No images were successfully processed")
                return
            
            # Create item for LLM queue if LLM analyzer is available
            if self.llm_analyzer:
                llm_item = {
                    "detection": detection,
                    "image_paths": image_paths,
                    "timestamp": time.time()
                }
                await self.llm_queue.put(llm_item)
            else:
                # If no LLM analyzer, send directly to UI queue
                await self.ui_queue.put(detection)
            
            # Update stats
            processing_time = (time.time() - start_time) * 1000  # ms
            self.stats["processing_times"]["image"].append(processing_time)
            logger.debug(f"Image processing completed in {processing_time:.2f} ms")
            
        except Exception as e:
            logger.error(f"Error processing images: {e}")
            self.stats["errors"] += 1
    
    async def _process_llm_analysis(self, item: Dict[str, Any]) -> None:
        """Process LLM analysis.
        
        Args:
            item: Queue item containing detection data and image paths.
        """
        start_time = time.time()
        
        try:
            detection = item["detection"]
            image_paths = item["image_paths"]
            
            logger.debug(f"Processing LLM analysis for: {type(detection).__name__}")
            
            if not self.llm_analyzer:
                logger.warning("LLM analyzer not available, skipping analysis")
                await self.ui_queue.put(detection)
                return
            
            # Analyze detection with LLM
            analyzed_detection = await self.llm_analyzer.analyze_detection(detection, image_paths)
            
            # Update stats
            self.stats["llm_analyses_performed"] += 1
            
            # Send to UI queue
            await self.ui_queue.put(analyzed_detection)
            
            # Update stats
            processing_time = (time.time() - start_time) * 1000  # ms
            self.stats["processing_times"]["llm"].append(processing_time)
            logger.debug(f"LLM analysis completed in {processing_time:.2f} ms")
            
        except Exception as e:
            logger.error(f"Error processing LLM analysis: {e}")
            self.stats["errors"] += 1
            
            # Forward original detection to UI queue on error
            if "detection" in item:
                await self.ui_queue.put(item["detection"])
    
    async def _process_ui_update(self, data: Union[DetectionData, ALPREventData]) -> None:
        """Process UI update.
        
        Args:
            data: Detection data with analysis results.
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Processing UI update for: {type(data).__name__}")
            
            # Call UI update callback if provided
            if self.ui_update_callback:
                await self.ui_update_callback(data)
                self.stats["ui_updates_sent"] += 1
            
            # Update stats
            processing_time = (time.time() - start_time) * 1000  # ms
            self.stats["processing_times"]["ui"].append(processing_time)
            logger.debug(f"UI update completed in {processing_time:.2f} ms")
            
        except Exception as e:
            logger.error(f"Error processing UI update: {e}")
            self.stats["errors"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue system statistics.
        
        Returns:
            Dict[str, Any]: Queue system statistics.
        """
        stats = self.stats.copy()
        
        # Calculate average processing times
        for queue_type, times in stats["processing_times"].items():
            if times:
                stats[f"avg_{queue_type}_time_ms"] = sum(times) / len(times)
            else:
                stats[f"avg_{queue_type}_time_ms"] = 0
        
        # Add queue sizes
        stats["queue_sizes"] = {
            "mqtt": self.mqtt_queue.size,
            "image": self.image_queue.size,
            "llm": self.llm_queue.size,
            "ui": self.ui_queue.size
        }
        
        # Add uptime
        if stats["start_time"]:
            stats["uptime_seconds"] = time.time() - stats["start_time"]
        
        return stats
