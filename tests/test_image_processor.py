"""Unit tests for the image processor module."""

import os
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
from PIL import Image
import io
import hashlib

from mqtt_client.processors.image import ImageProcessor
from mqtt_client.models.alpr_detection import ALPREventData, ImageInfoModel


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory for testing."""
    cache_dir = tmp_path / "image_cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def image_processor(temp_cache_dir):
    """Create an image processor instance for testing."""
    processor = ImageProcessor(cache_dir=temp_cache_dir, max_cache_size_mb=10)
    yield processor
    
    # Cleanup
    asyncio.run(processor.close())


@pytest.fixture
def mock_image_data():
    """Create a mock image for testing."""
    # Create a small test image in memory
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()


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


class TestImageProcessor:
    """Test cases for the ImageProcessor class."""

    @pytest.mark.asyncio
    async def test_download_image(self, image_processor, mock_image_data):
        """Test downloading an image."""
        test_url = "http://example.com/test.jpg"
        
        # Mock the aiohttp ClientSession
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read.return_value = mock_image_data
        
        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_response
        
        mock_client_session = AsyncMock()
        mock_client_session.get.return_value = mock_session
        
        # Patch the ClientSession creation
        with patch('aiohttp.ClientSession', return_value=mock_client_session):
            # Call the method
            result = await image_processor.download_image(test_url)
            
            # Verify the result
            assert result is not None
            assert os.path.exists(result)
            assert os.path.getsize(result) > 0
            
            # Verify the URL was called
            mock_client_session.get.assert_called_once_with(test_url)

    @pytest.mark.asyncio
    async def test_download_image_failure(self, image_processor):
        """Test downloading an image with a failure response."""
        test_url = "http://example.com/nonexistent.jpg"
        
        # Mock the aiohttp ClientSession with a 404 response
        mock_response = AsyncMock()
        mock_response.status = 404
        
        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_response
        
        mock_client_session = AsyncMock()
        mock_client_session.get.return_value = mock_session
        
        # Patch the ClientSession creation
        with patch('aiohttp.ClientSession', return_value=mock_client_session):
            # Call the method
            result = await image_processor.download_image(test_url)
            
            # Verify the result is None for a failed download
            assert result is None

    @pytest.mark.asyncio
    async def test_is_cached(self, image_processor, mock_image_data):
        """Test checking if an image is cached."""
        test_url = "http://example.com/cached.jpg"
        
        # Create a mock cached file
        url_hash = hashlib.md5(test_url.encode()).hexdigest()
        cache_path = os.path.join(image_processor._cache_dir, f"{url_hash}.jpg")
        
        # Write mock data to the cache file
        with open(cache_path, 'wb') as f:
            f.write(mock_image_data)
        
        # Update cache metadata
        image_processor._update_cache_metadata(test_url, cache_path)
        
        # Test the is_cached method
        assert image_processor._is_cached(test_url) is True
        assert image_processor._is_cached("http://example.com/not_cached.jpg") is False

    @pytest.mark.asyncio
    async def test_validate_image(self, image_processor, mock_image_data):
        """Test image validation."""
        # Create a temporary file with the mock image data
        temp_file = os.path.join(image_processor._cache_dir, "test_validate.jpg")
        with open(temp_file, 'wb') as f:
            f.write(mock_image_data)
        
        # Test validation
        result = await image_processor.validate_image(temp_file)
        assert result is not None
        assert isinstance(result, Image.Image)
        
        # Test validation with a non-image file
        bad_file = os.path.join(image_processor._cache_dir, "not_an_image.txt")
        with open(bad_file, 'w') as f:
            f.write("This is not an image")
        
        result = await image_processor.validate_image(bad_file)
        assert result is None

    @pytest.mark.asyncio
    async def test_preprocess_image(self, image_processor):
        """Test image preprocessing."""
        # Create a test image
        img = Image.new('RGB', (1920, 1080), color='blue')
        
        # Test preprocessing with max size
        processed = await image_processor.preprocess_image(img, max_size=(800, 600))
        assert processed.width <= 800
        assert processed.height <= 600
        
        # Test preprocessing without max size (should return original size)
        processed = await image_processor.preprocess_image(img)
        assert processed.width == 1920
        assert processed.height == 1080

    @pytest.mark.asyncio
    async def test_clean_cache(self, image_processor, mock_image_data):
        """Test cache cleaning functionality."""
        # Create multiple mock cached files to exceed the cache size limit
        for i in range(20):
            test_url = f"http://example.com/image{i}.jpg"
            url_hash = hashlib.md5(test_url.encode()).hexdigest()
            cache_path = os.path.join(image_processor._cache_dir, f"{url_hash}.jpg")
            
            # Create a large mock file (1MB each)
            with open(cache_path, 'wb') as f:
                f.write(mock_image_data * 1000)  # Make it large enough to trigger cleanup
            
            # Update cache metadata with staggered access times
            image_processor._update_cache_metadata(test_url, cache_path)
            # Simulate different last accessed times
            image_processor._cache_metadata[test_url]["last_accessed"] -= i
        
        # Set a small cache size limit to trigger cleanup
        image_processor._max_cache_size_bytes = 5 * 1024 * 1024  # 5MB
        
        # Run cache cleanup
        await image_processor._clean_cache()
        
        # Verify that some files were removed
        remaining_files = os.listdir(image_processor._cache_dir)
        assert len(remaining_files) < 20

    @pytest.mark.asyncio
    async def test_process_images_from_detection(self, image_processor, mock_alpr_data, mock_image_data):
        """Test processing images from detection data."""
        # Mock the download_image method
        async def mock_download(url):
            url_hash = hashlib.md5(url.encode()).hexdigest()
            cache_path = os.path.join(image_processor._cache_dir, f"{url_hash}.jpg")
            with open(cache_path, 'wb') as f:
                f.write(mock_image_data)
            return cache_path
        
        # Patch the necessary methods
        with patch.object(image_processor, 'download_image', side_effect=mock_download):
            with patch.object(image_processor, 'validate_image', return_value=Image.new('RGB', (100, 100))):
                # Call the method
                results = await image_processor.process_images_from_detection(mock_alpr_data)
                
                # Verify results
                assert len(results) == 2
                for url, result in results.items():
                    assert result is not None
                    assert len(result) == 2
                    assert os.path.exists(result[0])
                    assert isinstance(result[1], Image.Image)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
