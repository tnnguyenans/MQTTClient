"""Test script for the image download and processing feature."""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root directory to path to allow imports
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mqtt_client.config import load_config
from mqtt_client.processors.image import ImageProcessor
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


async def test_image_processor():
    """Test the image processor functionality."""
    # Load configuration
    config = load_config()
    
    # Create image processor
    image_processor = ImageProcessor(
        cache_dir=config.image_processor.cache_dir,
        max_cache_size_mb=config.image_processor.max_cache_size_mb
    )
    
    try:
        # Test 1: Download individual images
        logger.info("=== Test 1: Download Individual Images ===")
        for i, url in enumerate(TEST_IMAGES):
            logger.info(f"Downloading image {i+1}: {url}")
            cache_path = await image_processor.download_image(url)
            if cache_path:
                logger.info(f"✓ Successfully downloaded to: {cache_path}")
                logger.info(f"  File size: {os.path.getsize(cache_path) / 1024:.2f} KB")
            else:
                logger.error(f"✗ Failed to download image: {url}")
        
        # Test 2: Download multiple images at once
        logger.info("\n=== Test 2: Download Multiple Images ===")
        results = await image_processor.download_images(TEST_IMAGES)
        for url, path in results.items():
            if path:
                logger.info(f"✓ Downloaded: {url} -> {path}")
            else:
                logger.error(f"✗ Failed to download: {url}")
        
        # Test 3: Process images (download, validate, and preprocess)
        logger.info("\n=== Test 3: Process Images ===")
        processed_results = await image_processor.process_images(
            TEST_IMAGES, 
            max_size=config.image_processor.default_max_size
        )
        
        for url, result in processed_results.items():
            if result:
                cache_path, img = result
                logger.info(f"✓ Processed: {url}")
                logger.info(f"  Cache path: {cache_path}")
                logger.info(f"  Image dimensions: {img.width}x{img.height}")
            else:
                logger.error(f"✗ Failed to process: {url}")
        
        # Test 4: Process images from ALPR event data
        logger.info("\n=== Test 4: Process Images from ALPR Event Data ===")
        
        # Create a mock ALPR event data
        mock_alpr_data = ALPREventData(
            EventName="Test ALPR Event",
            CameraID=12345,
            CameraName="Test Camera",
            Detections=[],  # Empty for this test
            Images=[
                ImageInfoModel(ID=1, Image=TEST_IMAGES[0], ScaledFactor=1.0),
                ImageInfoModel(ID=2, Image=TEST_IMAGES[1], ScaledFactor=1.0),
            ]
        )
        
        # Process images from mock ALPR event data
        alpr_results = await image_processor.process_images_from_detection(
            mock_alpr_data,
            max_size=config.image_processor.default_max_size
        )
        
        for url, result in alpr_results.items():
            if result:
                cache_path, img = result
                logger.info(f"✓ Processed ALPR image: {url}")
                logger.info(f"  Cache path: {cache_path}")
                logger.info(f"  Image dimensions: {img.width}x{img.height}")
            else:
                logger.error(f"✗ Failed to process ALPR image: {url}")
        
        # Test 5: Verify cache directory
        logger.info("\n=== Test 5: Verify Cache Directory ===")
        cache_dir = config.image_processor.cache_dir
        logger.info(f"Cache directory: {cache_dir}")
        
        if os.path.exists(cache_dir):
            cache_files = os.listdir(cache_dir)
            logger.info(f"Found {len(cache_files)} files in cache directory")
            
            # Calculate total cache size
            total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in cache_files)
            logger.info(f"Total cache size: {total_size / (1024 * 1024):.2f} MB")
            
            # List some cached files
            for i, file in enumerate(cache_files[:5]):  # Show up to 5 files
                file_path = os.path.join(cache_dir, file)
                file_size = os.path.getsize(file_path)
                logger.info(f"  {i+1}. {file} ({file_size / 1024:.2f} KB)")
            
            if len(cache_files) > 5:
                logger.info(f"  ... and {len(cache_files) - 5} more files")
        else:
            logger.error(f"Cache directory does not exist: {cache_dir}")
    
    finally:
        # Clean up resources
        await image_processor.close()
        logger.info("Image processor closed")


async def test_main():
    """Main test function."""
    try:
        await test_image_processor()
    except Exception as e:
        logger.error(f"Error in test: {e}")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_main())
