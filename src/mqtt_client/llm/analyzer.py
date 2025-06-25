"""LLM-based image analyzer for extracting additional attributes from detected objects."""

import os
import logging
import asyncio
import base64
import json
import time
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
from pathlib import Path
import aiohttp
import io
from PIL import Image, ImageDraw

from mqtt_client.config import LLMConfig
from mqtt_client.models.detection import DetectionData, DetectedObject, BoundingBox
from mqtt_client.models.alpr_detection import ALPREventData
from mqtt_client.models.transformers import extract_bounding_boxes

# Configure logging
logger = logging.getLogger(__name__)


def truncate_image_data_for_logging(data: str, max_length: int = 30) -> str:
    """Truncate image data (URL or base64) for logging to avoid cluttering logs.
    
    Args:
        data: String data to truncate (URL or base64).
        max_length: Maximum length of the data to display.
        
    Returns:
        str: Truncated string data.
    """
    if not data:
        return ""
    
    # Handle data URI format with base64
    if data.startswith('data:') and 'base64,' in data:
        prefix, content = data.split('base64,', 1)
        return f"{prefix}base64,{content[:10]}...{content[-5:] if len(content) > 5 else ''}"
    
    # Check if it's a base64 image without prefix
    if len(data) > 100 and not data.startswith('http'):
        return f"{data[:10]}...{data[-5:] if len(data) > 5 else ''}"
        
    # For regular URLs, truncate if too long
    if data.startswith('http'):
        parts = data.split('/', 3)
        if len(parts) >= 4:
            return f"{parts[0]}//{parts[2]}/...{data[-10:] if len(data) > 10 else ''}"
    
    # Simple truncation for other data
    if len(data) > max_length:
        return f"{data[:max_length]}..."
        
    return data


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int, time_period: int = 60):
        """Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the time period.
            time_period: Time period in seconds (default: 60 seconds).
        """
        self.max_calls = max_calls
        self.time_period = time_period
        self.calls = []
    
    async def wait(self) -> None:
        """Wait until a call can be made without exceeding the rate limit."""
        now = time.time()
        
        # Remove calls older than the time period
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_period]
        
        # If we've reached the maximum number of calls, wait until the oldest call expires
        if len(self.calls) >= self.max_calls:
            wait_time = self.time_period - (now - self.calls[0])
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting for {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
                # Recursive call to check again after waiting
                return await self.wait()
        
        # Add the current call
        self.calls.append(time.time())


class LLMAnalyzer:
    """LLM-based image analyzer for extracting additional attributes from detected objects."""
    
    def __init__(self, config: LLMConfig):
        """Initialize LLM analyzer.
        
        Args:
            config: LLM configuration.
        """
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = RateLimiter(max_calls=config.rate_limit)
        
        logger.info(f"LLM analyzer initialized with provider: {config.provider}")
        logger.info(f"Using model: {config.model}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session.
        
        Returns:
            aiohttp.ClientSession: HTTP session.
        """
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self) -> None:
        """Close resources."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64.
        
        Args:
            image_path: Path to image file.
            
        Returns:
            str: Base64 encoded image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _crop_object_from_image(
        self, 
        image_path: str, 
        bbox: Union[BoundingBox, Dict[str, int]]
    ) -> Tuple[str, Image.Image]:
        """Crop object from image using bounding box.
        
        Args:
            image_path: Path to image file.
            bbox: Bounding box coordinates (either BoundingBox or Dict with Left, Top, Right, Bottom).
            
        Returns:
            Tuple[str, PIL.Image.Image]: Tuple of (output_path, cropped_image).
        """
        try:
            # Open the image
            img = Image.open(image_path)
            
            # Extract coordinates based on bbox type
            if isinstance(bbox, BoundingBox):
                # DetectionData format (x, y, width, height)
                x, y, width, height = bbox.x, bbox.y, bbox.width, bbox.height
                left = int(x)
                top = int(y)
                right = int(x + width)
                bottom = int(y + height)
            elif isinstance(bbox, dict):
                # Check for new format (x, y, width, height)
                if all(k in bbox for k in ["x", "y", "width", "height"]):
                    left = int(bbox["x"])
                    top = int(bbox["y"])
                    right = int(bbox["x"] + bbox["width"])
                    bottom = int(bbox["y"] + bbox["height"])
                # Check for old format (Left, Top, Right, Bottom)
                elif all(k in bbox for k in ["Left", "Top", "Right", "Bottom"]):
                    left = int(bbox["Left"])
                    top = int(bbox["Top"])
                    right = int(bbox["Right"])
                    bottom = int(bbox["Bottom"])
                else:
                    # Log available keys and raise error
                    logger.error(f"Invalid bounding box format. Available keys: {list(bbox.keys())}")
                    raise ValueError(f"Invalid bounding box format: {bbox}")
            else:
                # Unknown format
                logger.error(f"Unknown bounding box type: {type(bbox)}")
                raise ValueError(f"Unknown bounding box type: {type(bbox)}")
            
            # Ensure coordinates are within image bounds
            img_width, img_height = img.size
            left = max(0, left)
            top = max(0, top)
            right = min(img_width, right)
            bottom = min(img_height, bottom)
            
            # Crop the image
            cropped_img = img.crop((left, top, right, bottom))
            
            # Save cropped image to a temporary file using platform-independent path
            import tempfile
            import os
            
            # Create a temporary file with the correct extension
            fd, output_path = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)  # Close the file descriptor
            
            # Save the cropped image
            cropped_img.save(output_path, format='JPEG', quality=85)
            
            return output_path, cropped_img
            
        except Exception as e:
            logger.error(f"Error cropping object from image: {e}")
            # Return the original image if cropping fails
            return image_path, Image.open(image_path)
    
    def _visualize_bounding_boxes(
        self, 
        image_path: str, 
        bboxes: List[Dict[str, Any]]
    ) -> str:
        """Visualize bounding boxes on image.
        
        Args:
            image_path: Path to image file.
            bboxes: List of bounding boxes with coordinates and metadata.
            
        Returns:
            str: Path to image with bounding boxes.
        """
        try:
            # Open the image
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            
            # Draw each bounding box
            for bbox in bboxes:
                if "Left" in bbox and "Top" in bbox and "Right" in bbox and "Bottom" in bbox:
                    # ALPREventData format
                    left = int(bbox["Left"])
                    top = int(bbox["Top"])
                    right = int(bbox["Right"])
                    bottom = int(bbox["Bottom"])
                elif "x" in bbox and "y" in bbox and "width" in bbox and "height" in bbox:
                    # DetectionData format
                    left = int(bbox["x"])
                    top = int(bbox["y"])
                    right = int(bbox["x"] + bbox["width"])
                    bottom = int(bbox["y"] + bbox["height"])
                else:
                    continue
                
                # Draw rectangle
                draw.rectangle([(left, top), (right, bottom)], outline="red", width=2)
                
                # Draw label if available
                label = bbox.get("license_plate", "") or bbox.get("class_name", "")
                if label:
                    draw.text((left, top - 10), label, fill="red")
            
            # Save image with bounding boxes
            output_path = f"{image_path}_boxes.jpg"
            img.save(output_path, format='JPEG', quality=85)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error visualizing bounding boxes: {e}")
            return image_path
    
    async def _analyze_with_openai(
        self, 
        image_path: str, 
        prompt: str
    ) -> Optional[str]:
        """Analyze image with OpenAI API.
        
        Args:
            image_path: Path to image file.
            prompt: Prompt for image analysis.
            
        Returns:
            Optional[str]: Analysis result or None if analysis failed.
        """
        if not self.config.api_key:
            logger.error("OpenAI API key not provided")
            return None
        
        try:
            # Wait for rate limiting
            await self.rate_limiter.wait()
            
            # Encode image to base64
            base64_image = self._encode_image_to_base64(image_path)
            
            # Prepare request payload
            # Use the configured model or fall back to a default
            vision_model = self.config.model  # Use the configured model from settings
            logger.info(f"Using OpenAI vision model: {vision_model}")
            
            # Prepare payload for vision API
            payload = {
                "model": vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }
            
            # Log truncated base64 data for debugging
            logger.debug(f"Sending image data (truncated): data:image/jpeg;base64,{truncate_image_data_for_logging(base64_image)}")
            
            # Get session
            session = await self._get_session()
            
            # Set timeout for the request
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            # Set headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}"
            }
            
            # Log the API request with truncated data
            logger.debug(f"Sending request to OpenAI API with model: {vision_model}")
            
            # Make API request
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenAI API error: {response.status} - {error_text}")
                    return None
                
                # Parse response
                result = await response.json()
                
                # Extract content from response
                try:
                    content = result["choices"][0]["message"]["content"]
                    return content
                except (KeyError, IndexError) as e:
                    logger.error(f"Error parsing OpenAI response: {e}")
                    logger.debug(f"Response: {result}")
                    return None
                
        except aiohttp.ClientError as e:
            logger.error(f"OpenAI API request error: {e}")
            return None
        except asyncio.TimeoutError:
            logger.error(f"OpenAI API request timeout")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI API request: {e}")
            return None
    
    async def _analyze_with_ollama(
        self, 
        image_path: str, 
        prompt: str
    ) -> Optional[str]:
        """Analyze image with Ollama API.
        
        Args:
            image_path: Path to image file.
            prompt: Prompt for image analysis.
            
        Returns:
            Optional[str]: Analysis result or None if analysis failed.
        """
        try:
            # Wait for rate limiting
            await self.rate_limiter.wait()
            
            # Encode image to base64
            base64_image = self._encode_image_to_base64(image_path)
            
            # Prepare request payload
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
                "options": {
                    "temperature": self.config.temperature
                }
            }
            
            # Get session
            session = await self._get_session()
            
            # Set timeout for the request
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            # Make API request
            api_url = f"{self.config.ollama_base_url}/api/generate"
            async with session.post(
                api_url,
                json=payload,
                timeout=timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama API error: {response.status} - {error_text}")
                    return None
                
                # Parse response
                result = await response.json()
                
                # Extract content from response
                try:
                    content = result["response"]
                    return content
                except KeyError as e:
                    logger.error(f"Error parsing Ollama response: {e}")
                    logger.debug(f"Response: {result}")
                    return None
                
        except aiohttp.ClientError as e:
            logger.error(f"Ollama API request error: {e}")
            return None
        except asyncio.TimeoutError:
            logger.error(f"Ollama API request timeout")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Ollama API request: {e}")
            return None
    
    async def analyze_object(
        self, 
        image_path: str, 
        object_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a detected object in an image.
        
        Args:
            image_path: Path to image file.
            object_info: Object information including bounding box.
            
        Returns:
            Dict[str, Any]: Object information with analysis results.
        """
        try:
            # Get bounding box
            bbox = None
            if "bounding_box" in object_info:
                # Already in the right format
                bbox = object_info["bounding_box"]
            elif all(k in object_info for k in ["left", "top", "right", "bottom"]):
                # Convert to bounding box format
                bbox = {
                    "x": object_info["left"],
                    "y": object_info["top"],
                    "width": object_info["right"] - object_info["left"],
                    "height": object_info["bottom"] - object_info["top"]
                }
            
            if not bbox:
                logger.error("No bounding box found in object info")
                logger.debug(f"Object info keys: {list(object_info.keys())}")
                return object_info
            
            # Crop object from image
            cropped_path, _ = self._crop_object_from_image(image_path, bbox)
            
            # Prepare prompt based on object type
            object_type = object_info.get("class_name", "") or object_info.get("license_plate", "object")
            
            prompt = f"""
            Analyze this image showing a {object_type}.
            
            Please provide the following information:
            1. Color(s) of the {object_type}
            2. Any visible actions or state (moving, stationary, etc.)
            3. Brief description (5-10 words)
            4. Any other notable attributes
            
            Format your response as a JSON object with these keys: colors, actions, description, attributes
            """
            
            # Analyze with appropriate provider
            if self.config.provider == "openai":
                analysis_text = await self._analyze_with_openai(cropped_path, prompt)
            else:  # ollama
                analysis_text = await self._analyze_with_ollama(cropped_path, prompt)
            
            if not analysis_text:
                logger.warning(f"Failed to analyze {object_type}")
                return object_info
            
            # Parse analysis result
            try:
                # Try multiple approaches to extract JSON from the response
                analysis = None
                
                # First, try direct JSON parsing
                try:
                    analysis = json.loads(analysis_text)
                except json.JSONDecodeError:
                    # If that fails, try to extract JSON from markdown code block
                    import re
                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', analysis_text)
                    if json_match:
                        try:
                            analysis = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON from code block")
                    
                    # If that fails, try to find JSON-like structure with { }
                    if not analysis:
                        json_match = re.search(r'\{[\s\S]*?\}', analysis_text)
                        if json_match:
                            try:
                                # Clean up the extracted text - replace single quotes with double quotes
                                # and ensure property names are properly quoted
                                json_str = json_match.group(0)
                                # Replace unquoted property names with quoted ones
                                json_str = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', json_str)
                                # Replace single quotes with double quotes
                                json_str = json_str.replace("'", '"')
                                analysis = json.loads(json_str)
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse JSON from extracted structure")
                
                # If all parsing attempts fail, create a simple analysis from the text
                if not analysis:
                    logger.warning(f"Could not parse JSON from LLM response, using text as description")
                    # Extract key information using regex patterns
                    colors_match = re.search(r'[Cc]olors?[:\s]+(.*?)(?:\n|$)', analysis_text)
                    actions_match = re.search(r'[Aa]ctions?[:\s]+(.*?)(?:\n|$)', analysis_text)
                    desc_match = re.search(r'[Dd]escription[:\s]+(.*?)(?:\n|$)', analysis_text)
                    
                    colors = colors_match.group(1).strip() if colors_match else ""
                    actions = actions_match.group(1).strip() if actions_match else ""
                    description = desc_match.group(1).strip() if desc_match else analysis_text[:100]
                    
                    analysis = {
                        "description": description,
                        "colors": [colors] if colors else [],
                        "actions": [actions] if actions else [],
                        "attributes": []
                    }
                
                # Ensure all expected keys exist
                for key in ["colors", "actions", "description", "attributes"]:
                    if key not in analysis:
                        analysis[key] = [] if key != "description" else ""
                
                # Update object info with analysis
                object_info["llm_analysis"] = analysis
                
                # Log analysis
                logger.info(f"Analysis for {object_type}: {analysis}")
                
            except Exception as e:
                logger.error(f"Error parsing analysis result: {e}")
                logger.debug(f"Raw analysis: {analysis_text}")
            
            return object_info
            
        except Exception as e:
            logger.error(f"Error analyzing object: {e}")
            return object_info
    
    async def analyze_detection(
        self, 
        detection: Union[DetectionData, ALPREventData],
        image_paths: Dict[str, str]
    ) -> Union[DetectionData, ALPREventData]:
        """Analyze detection data.
        
        Args:
            detection: Detection data.
            image_paths: Dictionary mapping image sources to file paths.
            
        Returns:
            Union[DetectionData, ALPREventData]: Detection data with analysis results.
        """
        try:
            if isinstance(detection, DetectionData):
                # Process generic detection data
                if not detection.image_url or str(detection.image_url) not in image_paths:
                    logger.warning("No image URL or image not found in paths")
                    return detection
                
                # Get image path
                image_path = image_paths[str(detection.image_url)]
                
                # Analyze each object
                for i, obj in enumerate(detection.objects):
                    # Convert object to dict for analysis
                    object_info = {
                        "class_id": obj.class_id,
                        "class_name": obj.class_name,
                        "confidence": obj.confidence,
                        "bounding_box": {
                            "x": obj.bounding_box.x,
                            "y": obj.bounding_box.y,
                            "width": obj.bounding_box.width,
                            "height": obj.bounding_box.height
                        },
                        "object_id": obj.object_id
                    }
                    
                    # Analyze object
                    analyzed_info = await self.analyze_object(image_path, object_info)
                    
                    # Update object with analysis results
                    if "llm_analysis" in analyzed_info:
                        # We can't directly modify the Pydantic model, so we'll add it to metadata
                        if not detection.metadata:
                            detection.metadata = {}
                        
                        # Create or update object_analysis in metadata
                        if "object_analysis" not in detection.metadata:
                            detection.metadata["object_analysis"] = {}
                        
                        # Add analysis for this object
                        object_id = obj.object_id or f"obj_{i}"
                        detection.metadata["object_analysis"][object_id] = analyzed_info["llm_analysis"]
                
                return detection
                
            elif isinstance(detection, ALPREventData):
                # Process ALPR event data
                # Extract bounding boxes
                boxes = extract_bounding_boxes(detection)
                if not boxes:
                    logger.warning("No bounding boxes found in ALPR event data")
                    return detection
                
                # Get image paths
                if not detection.Images or len(detection.Images) == 0:
                    logger.warning("No images found in ALPR event data")
                    return detection
                
                if not detection.Images[0].Image:
                    logger.warning("Image URL is empty in ALPR event data")
                    return detection
                
                image_source = str(detection.Images[0].Image)
                if image_source not in image_paths:
                    logger.warning(f"Image not found in paths: {truncate_image_data_for_logging(image_source)}")
                    return detection
                
                image_path = image_paths[image_source]
                
                # Analyze each license plate
                for i, box in enumerate(boxes):
                    try:
                        # Analyze object
                        analyzed_info = await self.analyze_object(image_path, box)
                        
                        # Update detection data with analysis results
                        if "llm_analysis" in analyzed_info:
                            # Make sure we have valid detection data to update
                            if not detection.Detections or len(detection.Detections) == 0:
                                logger.warning("No Detections array in ALPR event data")
                                continue
                                
                            if not detection.Detections[0].DetectionData:
                                logger.warning("No DetectionData array in ALPR event data")
                                continue
                                
                            # Check if the index is valid
                            if i < len(detection.Detections[0].DetectionData):
                                detection_data = detection.Detections[0].DetectionData[i]
                                
                                # Convert analysis to JSON string
                                analysis_json = json.dumps(analyzed_info["llm_analysis"])
                                
                                # Update ExtraVisionResult
                                detection_data.ExtraVisionResult = analysis_json
                            else:
                                logger.warning(f"Detection data index {i} out of range (max: {len(detection.Detections[0].DetectionData) - 1})")
                    except IndexError as e:
                        logger.error(f"Index error updating detection data: {e}")
                    except Exception as e:
                        logger.error(f"Error analyzing box {i}: {e}")
                
                return detection
            
            else:
                logger.warning(f"Unsupported detection type: {type(detection).__name__}")
                return detection
                
        except Exception as e:
            logger.error(f"Error in analyze_detection: {e}")
            return detection