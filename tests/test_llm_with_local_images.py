"""Test LLM analyzer with local images."""

import os
import json
import pytest
import asyncio
import base64
from pathlib import Path
from unittest.mock import patch, AsyncMock

from mqtt_client.config import AppConfig, LLMConfig
from mqtt_client.models.detection import DetectionData, DetectedObject
from mqtt_client.llm.analyzer import LLMAnalyzer


@pytest.fixture
def llm_config():
    """Create a test LLM configuration."""
    return LLMConfig(
        provider="openai",
        openai_api_key=os.environ.get("OPENAI_API_KEY", "test_key"),
        openai_model="gpt-4-vision-preview",
        ollama_base_url="http://localhost:11434",
        ollama_model="llava",
        rate_limit_per_minute=10,
        max_retries=3
    )


@pytest.fixture
def app_config(llm_config):
    """Create a test application configuration."""
    return AppConfig(
        llm=llm_config,
        mqtt=None,
        image_processor=None
    )


def create_test_image(output_path):
    """Create a simple test image with colored shapes."""
    try:
        from PIL import Image, ImageDraw
        
        # Create a blank image with white background
        img = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw a red rectangle (simulating a car)
        draw.rectangle([(50, 50), (200, 150)], fill='red', outline='black')
        
        # Draw a blue circle (simulating a person)
        draw.ellipse([(250, 100), (350, 250)], fill='blue', outline='black')
        
        # Save the image
        img.save(output_path)
        return True
    except ImportError:
        print("PIL library not installed. Cannot create test image.")
        return False


@pytest.mark.asyncio
async def test_analyze_image_with_real_llm():
    """Test analyzing an image with the real LLM if API key is available."""
    # Skip test if no API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        pytest.skip("No valid OpenAI API key found in environment")
    
    # Create test directory if it doesn't exist
    test_dir = Path(__file__).parent / "data"
    test_dir.mkdir(exist_ok=True)
    
    # Create test image
    test_image_path = test_dir / "test_image.jpg"
    if not test_image_path.exists():
        if not create_test_image(test_image_path):
            pytest.skip("Could not create test image")
    
    # Create LLM config
    llm_config = LLMConfig(
        provider="openai",
        openai_api_key=api_key,
        openai_model="gpt-4-vision-preview",
        rate_limit_per_minute=10,
        max_retries=3
    )
    
    # Create detection data
    detection = DetectionData(
        source="test_camera",
        objects=[
            DetectedObject(
                id="obj1",
                label="rectangle",
                confidence=0.95,
                bounding_box={
                    "x": 50,
                    "y": 50,
                    "width": 150,
                    "height": 100
                }
            ),
            DetectedObject(
                id="obj2",
                label="circle",
                confidence=0.85,
                bounding_box={
                    "x": 250,
                    "y": 100,
                    "width": 100,
                    "height": 150
                }
            )
        ],
        image_url="file://local/test_image.jpg",
        additional_image_urls=[]
    )
    
    # Create image paths mapping
    image_paths = {
        "file://local/test_image.jpg": str(test_image_path)
    }
    
    # Create analyzer
    analyzer = LLMAnalyzer(config=llm_config)
    
    try:
        # Analyze detection
        result = await analyzer.analyze_detection(detection, image_paths)
        
        # Check that metadata was updated
        assert "object_analysis" in result.metadata
        assert len(result.metadata["object_analysis"]) > 0
        
        # Print analysis results for inspection
        print("\nLLM Analysis Results:")
        for obj_id, analysis in result.metadata["object_analysis"].items():
            print(f"Object {obj_id}:")
            if "description" in analysis:
                print(f"  Description: {analysis['description']}")
            if "colors" in analysis and analysis["colors"]:
                print(f"  Colors: {', '.join(analysis['colors'])}")
            if "actions" in analysis and analysis["actions"]:
                print(f"  Actions: {', '.join(analysis['actions'])}")
        
        # Basic validation of results
        for obj_id, analysis in result.metadata["object_analysis"].items():
            assert "description" in analysis
            assert isinstance(analysis.get("colors", []), list)
            assert isinstance(analysis.get("actions", []), list)
    
    finally:
        # Close analyzer
        await analyzer.close()


@pytest.mark.asyncio
async def test_analyze_image_with_ollama():
    """Test analyzing an image with Ollama if available."""
    # Skip test if Ollama is not configured
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Create test directory if it doesn't exist
    test_dir = Path(__file__).parent / "data"
    test_dir.mkdir(exist_ok=True)
    
    # Create test image
    test_image_path = test_dir / "test_image.jpg"
    if not test_image_path.exists():
        if not create_test_image(test_image_path):
            pytest.skip("Could not create test image")
    
    # Check if Ollama is running
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ollama_url}/api/tags") as response:
                if response.status != 200:
                    pytest.skip("Ollama server not available")
    except aiohttp.ClientError:
        pytest.skip("Ollama server not available")
    
    # Create LLM config
    llm_config = LLMConfig(
        provider="ollama",
        ollama_base_url=ollama_url,
        ollama_model="llava",
        rate_limit_per_minute=10,
        max_retries=3
    )
    
    # Create detection data
    detection = DetectionData(
        source="test_camera",
        objects=[
            DetectedObject(
                id="obj1",
                label="rectangle",
                confidence=0.95,
                bounding_box={
                    "x": 50,
                    "y": 50,
                    "width": 150,
                    "height": 100
                }
            ),
            DetectedObject(
                id="obj2",
                label="circle",
                confidence=0.85,
                bounding_box={
                    "x": 250,
                    "y": 100,
                    "width": 100,
                    "height": 150
                }
            )
        ],
        image_url="file://local/test_image.jpg",
        additional_image_urls=[]
    )
    
    # Create image paths mapping
    image_paths = {
        "file://local/test_image.jpg": str(test_image_path)
    }
    
    # Create analyzer
    analyzer = LLMAnalyzer(config=llm_config)
    
    try:
        # Analyze detection
        result = await analyzer.analyze_detection(detection, image_paths)
        
        # Check that metadata was updated
        assert "object_analysis" in result.metadata
        assert len(result.metadata["object_analysis"]) > 0
        
        # Print analysis results for inspection
        print("\nLLM Analysis Results (Ollama):")
        for obj_id, analysis in result.metadata["object_analysis"].items():
            print(f"Object {obj_id}:")
            if "description" in analysis:
                print(f"  Description: {analysis['description']}")
            if "colors" in analysis and analysis["colors"]:
                print(f"  Colors: {', '.join(analysis['colors'])}")
            if "actions" in analysis and analysis["actions"]:
                print(f"  Actions: {', '.join(analysis['actions'])}")
        
        # Basic validation of results
        for obj_id, analysis in result.metadata["object_analysis"].items():
            assert "description" in analysis
            assert isinstance(analysis.get("colors", []), list)
            assert isinstance(analysis.get("actions", []), list)
    
    finally:
        # Close analyzer
        await analyzer.close()