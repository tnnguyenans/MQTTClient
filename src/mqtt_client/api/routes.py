"""API routes for MQTT Client application."""

import os
import logging
import json
import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import FileResponse, RedirectResponse

from mqtt_client.config import load_config
from mqtt_client.models.detection import DetectionData
from mqtt_client.models.alpr_detection import ALPREventData
from mqtt_client.models.transformers import extract_bounding_boxes, map_alpr_to_detection_data
from mqtt_client.api.models import (
    DetectionEventModel,
    DetectionObjectModel,
    LicensePlateModel,
    BoundingBoxModel,
    HealthResponse,
    WebSocketMessage,
    ImageCacheInfo,
)
from mqtt_client.api.ws_manager import connection_manager

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global variables for detection history
detection_history: List[Dict[str, Any]] = []
MAX_HISTORY_SIZE = 100
start_time = time.time()

# Create detection event ID
def create_detection_id() -> str:
    """Create a unique detection event ID.
    
    Returns:
        str: Unique detection event ID.
    """
    return str(uuid.uuid4())


# Helper function to map detection data to API model
def map_detection_data_to_api_model(
    data: Union[DetectionData, ALPREventData]
) -> DetectionEventModel:
    """Map detection data to API model.
    
    Args:
        data: Detection data to map.
        
    Returns:
        DetectionEventModel: API model for detection event.
    """
    # Create unique event ID
    event_id = create_detection_id()
    
    if isinstance(data, DetectionData):
        # Map generic detection data
        objects = []
        
        for obj in data.objects:
            # Create detection object model
            detection_object = DetectionObjectModel(
                id=f"{event_id}_{obj.object_id}",
                class_name=obj.class_name,
                class_id=obj.class_id,
                confidence=obj.confidence,
                timestamp=obj.timestamp or datetime.utcnow(),
                bounding_box=BoundingBoxModel(
                    x=obj.bounding_box.x,
                    y=obj.bounding_box.y,
                    width=obj.bounding_box.width,
                    height=obj.bounding_box.height
                ),
                image_url=None  # Will be set later if available
            )
            
            # Add LLM analysis if available
            if data.metadata and "object_analysis" in data.metadata:
                obj_analysis = data.metadata["object_analysis"].get(str(obj.object_id))
                if obj_analysis:
                    detection_object.analysis = obj_analysis
            
            objects.append(detection_object)
        
        # Create event model
        event = DetectionEventModel(
            id=event_id,
            source=data.source or "unknown",
            timestamp=data.timestamp or datetime.utcnow(),
            objects=objects,
            image_url=str(data.image_url) if data.image_url else None,
            additional_image_urls=[str(url) for url in data.additional_image_urls],
            metadata=data.metadata or {}
        )
        
    elif isinstance(data, ALPREventData):
        # Map ALPR event data
        event_source = f"ALPR: {data.CameraName}" if data.CameraName else "ALPR"
        event_time = datetime.utcnow()  # Default timestamp
        
        # Extract timestamp from the first detection if available
        for detection_rule in data.Detections:
            for detection_data in detection_rule.DetectionData:
                if detection_data.Detections and detection_data.Detections[0].Time:
                    try:
                        event_time = datetime.fromisoformat(
                            detection_data.Detections[0].Time.replace('Z', '+00:00')
                        )
                        break
                    except (ValueError, TypeError):
                        pass
        
        # Extract bounding boxes for license plates
        boxes = extract_bounding_boxes(data)
        objects = []
        
        for box in boxes:
            # Create license plate model
            plate = LicensePlateModel(
                plate_number=box["license_plate"],
                confidence=box["confidence"],
                timestamp=box["timestamp"],
                bounding_box=BoundingBoxModel(
                    x=box["bounding_box"]["x"],
                    y=box["bounding_box"]["y"],
                    width=box["bounding_box"]["width"],
                    height=box["bounding_box"]["height"]
                ),
                image_url=None  # Will be set later if available
            )
            
            # Add LLM analysis if available
            for detection_rule in data.Detections:
                for detection_data in detection_rule.DetectionData:
                    if (detection_data.Name == box["license_plate"] and 
                            detection_data.ExtraVisionResult):
                        try:
                            analysis = json.loads(detection_data.ExtraVisionResult)
                            plate.analysis = analysis
                        except json.JSONDecodeError:
                            pass
            
            objects.append(plate)
        
        # Create event model from ALPR data
        event = DetectionEventModel(
            id=event_id,
            source=event_source,
            timestamp=event_time,
            objects=objects,
            image_url=None,  # Will be set later if available
            metadata={"camera_id": data.CameraID} if data.CameraID else {}
        )
        
        # Add image URLs if available
        if data.Images:
            event.image_url = data.Images[0].Image
            event.additional_image_urls = [img.Image for img in data.Images[1:]]
            
    else:
        raise ValueError(f"Unsupported detection data type: {type(data).__name__}")
    
    return event


@router.get("/", summary="API root")
async def root() -> Dict[str, str]:
    """API root endpoint.
    
    Returns:
        Dict[str, str]: Welcome message.
    """
    return {"message": "MQTT Client API"}


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check() -> HealthResponse:
    """Health check endpoint.
    
    Returns:
        HealthResponse: Service health status.
    """
    # Get application config
    config = load_config()
    
    # Calculate uptime
    uptime = time.time() - start_time
    
    # Create components status
    components = {
        "mqtt": {
            "status": "up",  # Should be dynamic based on MQTT client status
            "broker": config.mqtt.broker_host,
        },
        "image_processor": {
            "status": "up",  # Should be dynamic based on image processor status
            "cache_dir": config.image_processor.cache_dir,
        },
        "llm": {
            "status": "up",  # Should be dynamic based on LLM analyzer status
            "provider": config.llm.provider,
        },
    }
    
    return HealthResponse(
        status="ok",
        version=config.version,
        uptime=uptime,
        components=components
    )


@router.get("/detections", response_model=List[DetectionEventModel], summary="Get detection history")
async def get_detections(limit: int = 10, skip: int = 0) -> List[DetectionEventModel]:
    """Get detection history.
    
    Args:
        limit: Maximum number of items to return.
        skip: Number of items to skip.
        
    Returns:
        List[DetectionEventModel]: List of detection events.
    """
    # Apply pagination
    start_idx = min(skip, len(detection_history))
    end_idx = min(start_idx + limit, len(detection_history))
    
    # Return paginated history (most recent first)
    return detection_history[start_idx:end_idx]


@router.get("/detections/{detection_id}", response_model=DetectionEventModel, summary="Get detection by ID")
async def get_detection(detection_id: str) -> DetectionEventModel:
    """Get detection by ID.
    
    Args:
        detection_id: Detection ID.
        
    Returns:
        DetectionEventModel: Detection event.
        
    Raises:
        HTTPException: If detection not found.
    """
    # Find detection by ID
    for detection in detection_history:
        if detection.id == detection_id:
            return detection
    
    # Return 404 if not found
    raise HTTPException(status_code=404, detail="Detection not found")


@router.get("/images/cache", response_model=ImageCacheInfo, summary="Get image cache info")
async def get_image_cache_info() -> ImageCacheInfo:
    """Get image cache information.
    
    Returns:
        ImageCacheInfo: Current image cache information.
    """
    # Get application config
    config = load_config()
    
    # Get image processor cache directory
    cache_dir = Path(config.image_processor.cache_dir)
    
    # Calculate cache size
    cache_size = sum(f.stat().st_size for f in cache_dir.glob('**/*') if f.is_file())
    
    # Count files
    item_count = sum(1 for _ in cache_dir.glob('**/*') if _.is_file())
    
    return ImageCacheInfo(
        cache_size=cache_size,
        max_cache_size=config.image_processor.max_cache_size_mb * 1024 * 1024,
        item_count=item_count,
        directory=str(cache_dir.absolute())
    )


@router.get("/images/{filename}", summary="Get image by filename")
async def get_image(filename: str) -> FileResponse:
    """Get image by filename.
    
    Args:
        filename: Image filename.
        
    Returns:
        FileResponse: Image file.
        
    Raises:
        HTTPException: If image not found.
    """
    # Get application config
    config = load_config()
    
    # Check if this might be a base64 string (they sometimes get passed as filenames)
    if len(filename) > 100 and not filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
        # Process as raw image data using our image endpoint
        logger.info(f"Processing potential raw image data with image endpoint")
        # Use our image processing function directly
        try:
            # Create a request model with the filename as data
            request = ImageRequestModel(data=filename)
            # Process the image data
            return await get_image_data(request)
        except Exception as e:
            logger.error(f"Error processing raw image data: {str(e)}")
            # Create a response with JavaScript that will POST to our image endpoint as fallback
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Image Redirect</title>
                <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    // Get the image data from the URL
                    const imageData = '{filename}';
                    
                    // Post to our image endpoint
                    fetch('/api/image', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{ data: imageData }})
                    }})
                    .then(response => response.blob())
                    .then(blob => {{
                        // Create an object URL for the blob
                        const url = URL.createObjectURL(blob);
                        
                        // Create an image element
                        const img = document.createElement('img');
                        img.src = url;
                        img.style.maxWidth = '100%';
                        
                        // Add the image to the page
                        document.body.appendChild(img);
                    }})
                    .catch(error => {{
                        console.error('Error:', error);
                        document.body.innerHTML = '<p>Error loading image</p>';
                    }});
                }});
                </script>
            </head>
            <body>
                <p>Loading image...</p>
            </body>
            </html>
            """
            return Response(content=html_content, media_type="text/html")
    
    # Get image path
    image_path = Path(config.image_processor.cache_dir) / filename
    
    # Check if file exists
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Return image file
    return FileResponse(
        image_path, 
        media_type=f"image/{image_path.suffix.replace('.', '')}"
    )


class ImageRequestModel(BaseModel):
    """Model for image request data."""
    data: str

@router.get("/base64/{encoded_data:path}", summary="Legacy endpoint for base64 images")
async def legacy_base64_image(encoded_data: str) -> Response:
    """Legacy endpoint for base64 images.
    
    This endpoint handles requests to the old /base64/ path for backward compatibility.
    
    Args:
        encoded_data: Base64 encoded image data from URL path.
        
    Returns:
        Response: Image data as JPEG.
    """
    logger.info(f"Received request to legacy base64 endpoint with data length: {len(encoded_data) if encoded_data else 0} chars")
    # Create a request model with the encoded data
    request = ImageRequestModel(data=encoded_data)
    # Process the image data using our main image processing function
    return await get_image_data(request)

@router.post("/api/image", summary="Get image from raw data or base64")
async def get_image_data(request: ImageRequestModel) -> Response:
    """Process image data and return it.
    
    Args:
        request: Request model containing image data.
        
    Returns:
        Response: Image data as JPEG.
        
    Raises:
        HTTPException: If data cannot be processed.
    """
    encoded_data = request.data
    logger.info(f"Received image request with data length: {len(encoded_data) if encoded_data else 0} chars")
    
    if not encoded_data:
        logger.error("No image data provided")
        raise HTTPException(status_code=400, detail="No image data provided")
    
    try:
        # Import required modules
        import urllib.parse
        from io import BytesIO
        from PIL import Image
        import base64
        import re
        
        # URL decode the string first (it may have been URL encoded)
        decoded_data = urllib.parse.unquote(encoded_data)
        logger.debug(f"URL decoded data length: {len(decoded_data)}")
        
        # Log the first and last few characters to help debug format issues
        if len(decoded_data) > 20:
            prefix = decoded_data[:20]
            suffix = decoded_data[-20:]
            logger.debug(f"Data prefix: {prefix}... suffix: ...{suffix}")
        
        # Initialize image_data variable
        image_data = None
        
        # Check if this is a data URI format
        if decoded_data.startswith('data:'):
            logger.info("Detected data URI format")
            # Extract the content type and data part
            header, data_part = decoded_data.split(',', 1)
            content_type = header.split(';')[0].split(':')[1]
            logger.info(f"Content type from URI: {content_type}")
            
            # Check if this is base64 encoded
            if ';base64' in header:
                logger.info("Data URI contains base64 encoded data")
                # Decode base64 data
                try:
                    image_data = base64.b64decode(data_part)
                    logger.info(f"Successfully decoded base64 data to {len(image_data)} bytes")
                except Exception as b64_error:
                    logger.error(f"Failed to decode base64 data: {str(b64_error)}")
                    raise HTTPException(status_code=400, detail=f"Invalid base64 data in URI: {str(b64_error)}")
            else:
                # Raw data in URI
                logger.info("Data URI contains raw data")
                image_data = data_part.encode('latin1')  # Use latin1 to preserve byte values
        
        # Check if it's already base64 encoded (without data URI prefix)
        elif re.match(r'^[A-Za-z0-9+/]+={0,2}$', decoded_data):
            logger.info("Detected base64 encoded data without URI prefix")
            try:
                # Try to decode as base64
                # Add padding if needed
                padding_needed = len(decoded_data) % 4
                if padding_needed:
                    decoded_data += '=' * (4 - padding_needed)
                
                image_data = base64.b64decode(decoded_data)
                logger.info(f"Successfully decoded base64 data to {len(image_data)} bytes")
            except Exception as b64_error:
                logger.error(f"Failed to decode as base64: {str(b64_error)}")
                # Continue to next approach
                image_data = None
        
        # If not base64, assume it's raw JPEG data
        if image_data is None:
            logger.info("Processing as raw JPEG string data")
            try:
                # Convert string to bytes using latin1 encoding to preserve byte values
                image_data = decoded_data.encode('latin1')
                logger.info(f"Converted string to {len(image_data)} bytes using latin1 encoding")
            except Exception as enc_error:
                logger.error(f"Failed to encode string as bytes: {str(enc_error)}")
                raise HTTPException(status_code=400, detail=f"Failed to process image data: {str(enc_error)}")
        
        # Validate the image data by trying to open it with PIL
        try:
            img = Image.open(BytesIO(image_data))
            logger.info(f"Validated image: {img.format}, size: {img.size}, mode: {img.mode}")
            
            # Convert to JPEG if it's not already
            if img.format != 'JPEG':
                logger.info(f"Converting {img.format} to JPEG")
                output = BytesIO()
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output, format='JPEG', quality=95)
                image_data = output.getvalue()
                logger.info(f"Converted to JPEG: {len(image_data)} bytes")
        except Exception as img_error:
            logger.error(f"Invalid image data: {str(img_error)}")
            # Try one more approach - clean the data and try base64 decoding
            try:
                # Clean the string to only include valid base64 characters
                cleaned_data = re.sub(r'[^A-Za-z0-9+/=]', '', decoded_data)
                # Add padding if needed
                padding_needed = len(cleaned_data) % 4
                if padding_needed:
                    cleaned_data += '=' * (4 - padding_needed)
                    
                image_data = base64.b64decode(cleaned_data)
                logger.info(f"Decoded after cleaning: {len(image_data)} bytes")
                
                # Verify it's a valid image
                img = Image.open(BytesIO(image_data))
                logger.info(f"Validated cleaned image: {img.format}, size: {img.size}")
            except Exception as final_error:
                logger.error(f"All image processing attempts failed: {str(final_error)}")
                # Continue with the original data, let the browser try to render it
        
        # Return the image
        return Response(
            content=image_data,
            media_type="image/jpeg"
        )
    except Exception as e:
        logger.error(f"Error processing image data: {str(e)}")
        # Log the first 100 characters of input to help debug
        if encoded_data and len(encoded_data) > 0:
            sample = encoded_data[:min(100, len(encoded_data))]
            logger.error(f"Sample of problematic data: {sample}...")
        raise HTTPException(status_code=400, detail=f"Could not process image data: {str(e)}")



@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time updates.
    
    Args:
        websocket: WebSocket connection.
    """
    client_id = await connection_manager.connect(websocket)
    
    try:
        # Send initial data (last 5 detections)
        if detection_history:
            recent_detections = detection_history[:5]
            await connection_manager.send_personal_message({
                "type": "initial_data",
                "data": {
                    "detections": [detection.dict() for detection in recent_detections]
                }
            }, websocket)
        
        # Wait for messages (keep connection alive)
        while True:
            # Receive and acknowledge client messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Acknowledge receipt of message
                await connection_manager.send_personal_message({
                    "type": "ack",
                    "data": {"client_id": client_id}
                }, websocket)
                
                # Handle client messages here if needed
                
            except json.JSONDecodeError:
                # Send error for invalid JSON
                await connection_manager.send_personal_message({
                    "type": "error",
                    "data": {"message": "Invalid JSON"}
                }, websocket)
                
    except WebSocketDisconnect:
        # Handle disconnection
        connection_manager.disconnect(websocket)
    except Exception as e:
        # Handle other exceptions
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)


async def handle_detection_update(data: Union[DetectionData, ALPREventData]) -> None:
    """Handle detection update for API.
    
    This function is called when a new detection is processed.
    It updates the detection history and broadcasts to WebSocket clients.
    
    Args:
        data: Detection data.
    """
    try:
        # Map detection data to API model
        detection_event = map_detection_data_to_api_model(data)
        
        # Add to detection history (limited size)
        detection_history.insert(0, detection_event)
        if len(detection_history) > MAX_HISTORY_SIZE:
            detection_history.pop()
        
        # Broadcast to WebSocket clients
        await connection_manager.broadcast({
            "type": "detection",
            "data": detection_event.dict()
        })
        
        logger.info(f"Detection update broadcast to {len(connection_manager.active_connections)} clients")
        
    except Exception as e:
        logger.error(f"Error handling detection update: {e}")