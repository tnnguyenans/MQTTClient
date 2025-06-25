"""Queue manager for asynchronous processing of MQTT messages and images."""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, Awaitable, TypeVar, Generic, Union, List

from mqtt_client.models.detection import DetectionData
from mqtt_client.models.alpr_detection import ALPREventData

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for queue items
T = TypeVar('T')


class ProcessingQueue(Generic[T]):
    """Generic processing queue for asynchronous processing."""
    
    def __init__(
        self, 
        name: str,
        max_size: int = 100,
        processor: Optional[Callable[[T], Awaitable[Any]]] = None
    ):
        """Initialize processing queue.
        
        Args:
            name: Queue name for logging.
            max_size: Maximum queue size.
            processor: Async function to process queue items.
        """
        self.name = name
        self.queue: asyncio.Queue[T] = asyncio.Queue(maxsize=max_size)
        self.processor = processor
        self.running = False
        self._task: Optional[asyncio.Task] = None
        
        logger.info(f"Created processing queue: {name} (max_size={max_size})")
    
    async def put(self, item: T) -> None:
        """Add item to queue.
        
        Args:
            item: Item to add to queue.
        """
        try:
            await self.queue.put(item)
            logger.debug(f"Added item to queue {self.name} (size={self.queue.qsize()})")
        except asyncio.QueueFull:
            logger.warning(f"Queue {self.name} is full, dropping item")
    
    async def get(self) -> T:
        """Get item from queue.
        
        Returns:
            T: Queue item.
        """
        return await self.queue.get()
    
    def task_done(self) -> None:
        """Mark task as done."""
        self.queue.task_done()
    
    async def process_queue(self) -> None:
        """Process queue items."""
        if not self.processor:
            logger.error(f"No processor defined for queue {self.name}")
            return
        
        logger.info(f"Started processing queue: {self.name}")
        
        while self.running:
            try:
                # Get item from queue
                item = await self.queue.get()
                
                try:
                    # Process item
                    await self.processor(item)
                except Exception as e:
                    logger.error(f"Error processing item in queue {self.name}: {e}")
                finally:
                    # Mark task as done
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                logger.info(f"Queue processor for {self.name} was cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in queue {self.name}: {e}")
    
    async def start(self) -> None:
        """Start queue processor.
        
        This is an async function to allow proper awaiting in the main application.
        """
        if self.running:
            logger.warning(f"Queue processor for {self.name} is already running")
            return
        
        self.running = True
        self._task = asyncio.create_task(self.process_queue())
        logger.info(f"Started queue processor for {self.name}")
    
    async def stop(self) -> None:
        """Stop queue processor."""
        if not self.running:
            logger.warning(f"Queue processor for {self.name} is not running")
            return
        
        self.running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        logger.info(f"Stopped queue processor for {self.name}")
    
    @property
    def size(self) -> int:
        """Get current queue size.
        
        Returns:
            int: Current queue size.
        """
        return self.queue.qsize()
    
    @property
    def is_empty(self) -> bool:
        """Check if queue is empty.
        
        Returns:
            bool: True if queue is empty, False otherwise.
        """
        return self.queue.empty()
    
    @property
    def is_full(self) -> bool:
        """Check if queue is full.
        
        Returns:
            bool: True if queue is full, False otherwise.
        """
        return self.queue.full()


class ImageProcessingQueue(ProcessingQueue[Union[DetectionData, ALPREventData]]):
    """Queue for processing images from detection data."""
    
    def __init__(
        self,
        processor: Callable[[Union[DetectionData, ALPREventData]], Awaitable[Any]],
        max_size: int = 100
    ):
        """Initialize image processing queue.
        
        Args:
            processor: Async function to process detection data.
            max_size: Maximum queue size.
        """
        super().__init__(name="image_processing", max_size=max_size, processor=processor)
