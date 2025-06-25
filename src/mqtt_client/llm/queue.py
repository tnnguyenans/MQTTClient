"""LLM processing queue for asynchronous image analysis."""

import asyncio
import logging
import time
from typing import Dict, Optional, Union, Any, Callable, Awaitable

from mqtt_client.models.detection import DetectionData
from mqtt_client.models.alpr_detection import ALPREventData
from mqtt_client.processors.queue_manager import ProcessingQueue
from mqtt_client.llm.analyzer import LLMAnalyzer

# Configure logging
logger = logging.getLogger(__name__)


class LLMProcessingQueue(ProcessingQueue[Union[Dict[str, Any], DetectionData, ALPREventData]]):
    """Queue for processing images with LLM analysis."""
    
    def __init__(
        self,
        analyzer: LLMAnalyzer,
        callback: Optional[Callable[[Union[DetectionData, ALPREventData]], Awaitable[None]]] = None,
        max_size: int = 50
    ):
        """Initialize LLM processing queue.
        
        Args:
            analyzer: LLM analyzer instance.
            callback: Optional callback function to call after processing.
            max_size: Maximum queue size.
        """
        super().__init__(name="llm_processing", max_size=max_size)
        self.analyzer = analyzer
        self.callback = callback
        self.processor = self._process_item
        
        logger.info("LLM processing queue initialized")
    
    async def _process_item(
        self, 
        item: Union[Dict[str, Any], DetectionData, ALPREventData]
    ) -> None:
        """Process queue item.
        
        Args:
            item: Queue item containing detection data and image paths.
        """
        try:
            start_time = time.time()
            
            # Extract detection and image paths from item
            if isinstance(item, dict):
                detection = item.get("detection")
                image_paths = item.get("image_paths", {})
            else:
                # If item is directly a detection, we don't have image paths
                detection = item
                image_paths = {}
            
            if not detection:
                logger.warning("No detection data in queue item")
                return
            
            # Skip if no image paths
            if not image_paths:
                logger.warning("No image paths in queue item, skipping LLM analysis")
                return
            
            # Analyze detection
            logger.info(f"Processing detection with LLM: {type(detection).__name__}")
            
            # Analyze detection with LLM
            analyzed_detection = await self.analyzer.analyze_detection(detection, image_paths)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # ms
            logger.info(f"LLM analysis completed in {processing_time:.2f} ms")
            
            # Call callback if provided
            if self.callback:
                try:
                    await self.callback(analyzed_detection)
                except Exception as e:
                    logger.error(f"Error in LLM processing callback: {e}")
            
        except Exception as e:
            logger.error(f"Error processing item in LLM queue: {e}")
    
    async def put_detection(
        self, 
        detection: Union[DetectionData, ALPREventData],
        image_paths: Dict[str, str]
    ) -> None:
        """Add detection to queue.
        
        Args:
            detection: Detection data.
            image_paths: Dictionary mapping image sources to file paths.
        """
        item = {
            "detection": detection,
            "image_paths": image_paths
        }
        await self.put(item)