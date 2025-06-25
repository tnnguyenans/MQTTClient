"""
Test script for LLM integration feature.

This script demonstrates how to use the LLM analyzer with a sample image.
"""

import asyncio
import logging
import os
import sys
import json
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Pillow not installed. Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
    from PIL import Image, ImageDraw

from mqtt_client.config import load_config
from mqtt_client.models.detection import DetectionData, DetectedObject
from mqtt_client.llm.analyzer import LLMAnalyzer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_test_image(output_path, width=800, height=600):
    """Create a simple test image with colored shapes."""
    # Create a blank image with white background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a red car
    draw.rectangle([(50, 100), (350, 250)], fill='red', outline='black', width=2)
    
    # Draw a blue person
    # Head
    draw.ellipse([(500, 100), (550, 150)], fill='blue', outline='black', width=2)
    # Body
    draw.rectangle([(510, 150), (540, 250)], fill='blue', outline='black', width=2)
    # Legs
    draw.rectangle([(510, 250), (520, 350)], fill='blue', outline='black', width=2)
    draw.rectangle([(530, 250), (540, 350)], fill='blue', outline='black', width=2)
    # Arms
    draw.rectangle([(480, 170), (510, 180)], fill='blue', outline='black', width=2)
    draw.rectangle([(540, 170), (570, 180)], fill='blue', outline='black', width=2)
    
    # Save the image
    img.save(output_path)
    logger.info(f"Created test image at {output_path}")
    return output_path


async def test_llm_analyzer():
    """Test the LLM analyzer with a sample image."""
    # Load configuration
    config = load_config()
    
    # Check if OpenAI API key is set
    if config.llm.provider == "openai" and (not config.llm.api_key or config.llm.api_key == "your_openai_api_key_here"):
        logger.error("OpenAI API key not set. Please update your .env file with a valid API key.")
        logger.info("Example .env configuration:")
        logger.info("LLM_PROVIDER=openai")
        logger.info("OPENAI_API_KEY=sk-your_actual_key_here")
        return
    
    # Check if using Ollama and verify it's running
    if config.llm.provider == "ollama":
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f"{config.llm.ollama_base_url}/api/tags", timeout=5) as response:
                        if response.status != 200:
                            logger.error(f"Ollama server not available at {config.llm.ollama_base_url}")
                            logger.info("Please make sure Ollama is running and the base URL is correct.")
                            return
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    logger.error(f"Could not connect to Ollama server at {config.llm.ollama_base_url}")
                    logger.info("Please make sure Ollama is running and the base URL is correct.")
                    return
        except ImportError:
            logger.error("aiohttp not installed. Installing now...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp"])
            logger.info("Please run the script again.")
            return
    
    # Create test directory if it doesn't exist
    test_dir = Path("./test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Create test image
    test_image_path = test_dir / "test_objects.jpg"
    create_test_image(test_image_path)
    
    # Create detection data
    detection = DetectionData(
        source="test_camera",
        objects=[
            DetectedObject(
                id="obj1",
                label="car",
                confidence=0.95,
                bounding_box={
                    "x": 50,
                    "y": 100,
                    "width": 300,
                    "height": 150
                }
            ),
            DetectedObject(
                id="obj2",
                label="person",
                confidence=0.85,
                bounding_box={
                    "x": 500,
                    "y": 100,
                    "width": 70,
                    "height": 250
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
    analyzer = LLMAnalyzer(config=config.llm)
    
    try:
        logger.info(f"Using LLM provider: {config.llm.provider}")
        if config.llm.provider == "openai":
            logger.info(f"Using OpenAI model: {config.llm.model}")
        else:
            logger.info(f"Using Ollama model: {config.llm.ollama_model}")
        
        # Analyze detection
        logger.info("Sending image to LLM for analysis...")
        try:
            result = await analyzer.analyze_detection(detection, image_paths)
        except Exception as e:
            logger.error(f"Error during LLM analysis: {e}")
            logger.info("Check your API key and network connection.")
            return
        
        # Check that metadata was updated
        if "object_analysis" in result.metadata:
            logger.info("Analysis successful!")
            
            # Print analysis results
            logger.info("\nLLM Analysis Results:")
            for obj_id, analysis in result.metadata["object_analysis"].items():
                logger.info(f"Object {obj_id} ({detection.objects[int(obj_id)-1].label}):")
                if "description" in analysis:
                    logger.info(f"  Description: {analysis['description']}")
                if "colors" in analysis and analysis["colors"]:
                    logger.info(f"  Colors: {', '.join(analysis['colors'])}")
                if "actions" in analysis and analysis["actions"]:
                    logger.info(f"  Actions: {', '.join(analysis['actions'])}")
        else:
            logger.error("Analysis failed - no object_analysis in metadata")
    
    finally:
        # Close analyzer
        await analyzer.close()


if __name__ == "__main__":
    try:
        asyncio.run(test_llm_analyzer())
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError running test: {e}")
        import traceback
        traceback.print_exc()