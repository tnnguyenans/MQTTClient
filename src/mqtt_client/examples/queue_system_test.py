"""Test script for the queue system feature."""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root directory to path to allow imports
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mqtt_client.config import load_config
from mqtt_client.processors.image import ImageProcessor
from mqtt_client.processors.queue_system import QueueSystem
from mqtt_client.llm.analyzer import LLMAnalyzer
from mqtt_client.models.alpr_detection import ALPREventData, ImageInfoModel
from mqtt_client.models.detection import DetectionData, DetectedObject, BoundingBox

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


async def handle_ui_update(data):
    """Handle UI update."""
    logger.info(f"UI Update received for: {type(data).__name__}")
    
    if isinstance(data, DetectionData):
        # Log analysis results from metadata
        if data.metadata and "object_analysis" in data.metadata:
            analyses = data.metadata["object_analysis"]
            logger.info(f"Analysis for {len(analyses)} objects")
            
            for obj_id, analysis in analyses.items():
                logger.info(f"Analysis for object {obj_id}:")
                if "description" in analysis:
                    logger.info(f"  Description: {analysis['description']}")
                if "colors" in analysis and analysis["colors"]:
                    logger.info(f"  Colors: {', '.join(analysis['colors'])}")
                if "actions" in analysis and analysis["actions"]:
                    logger.info(f"  Actions: {', '.join(analysis['actions'])}")
    
    elif isinstance(data, ALPREventData):
        # Log analysis results from ExtraVisionResult
        for detection_rule in data.Detections:
            for detection_data in detection_rule.DetectionData:
                if detection_data.ExtraVisionResult:
                    try:
                        import json
                        analysis = json.loads(detection_data.ExtraVisionResult)
                        
                        logger.info(f"Analysis for license plate {detection_data.Name}:")
                        if "description" in analysis:
                            logger.info(f"  Description: {analysis['description']}")
                        if "colors" in analysis and analysis["colors"]:
                            logger.info(f"  Colors: {', '.join(analysis['colors'])}")
                        if "actions" in analysis and analysis["actions"]:
                            logger.info(f"  Actions: {', '.join(analysis['actions'])}")
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in ExtraVisionResult: {detection_data.ExtraVisionResult}")


def create_sample_detection_data():
    """Create sample detection data."""
    return DetectionData(
        source="test_source",
        timestamp="2023-01-01T12:00:00Z",
        image_url=TEST_IMAGES[0],
        objects=[
            DetectedObject(
                object_id="1",
                class_id=1,
                class_name="car",
                confidence=0.95,
                bounding_box=BoundingBox(x=100, y=100, width=200, height=150)
            )
        ]
    )


def create_sample_alpr_event_data():
    """Create sample ALPR event data."""
    from mqtt_client.models.alpr_detection import (
        RuleModel, 
        DetectionDataModel, 
        ModelClassIDModel,
        ModelClassNameModel,
        DetectionItemModel,
        BoundingBoxModel,
        UserInfoModel
    )
    
    return ALPREventData(
        EventName="ALPR Test Event",
        CameraID=12345,
        CameraName="Test Camera",
        Detections=[
            RuleModel(
                RuleType="Presence",
                DetectionData=[
                    DetectionDataModel(
                        Name="ABC123",
                        ModelClassID=ModelClassIDModel(ModelID=1, Class=1),
                        ModelClassName=ModelClassNameModel(Model="LPR", Class=""),
                        DeploymentGroupID=1,
                        DeploymentGroupName="Test Group",
                        CameraID=12345,
                        CameraName="Test Camera",
                        CameraAddress="test.mp4",
                        TrackID=1,
                        Detections=[
                            DetectionItemModel(
                                Time="2023-01-01T12:00:00Z",
                                BoundingBox=BoundingBoxModel(Left=100, Top=100, Right=200, Bottom=150),
                                Score=0.95,
                                Image=1,
                                ModelClassID=ModelClassIDModel(ModelID=1, Class=1),
                                ModelClassName=ModelClassNameModel(Model="LPR", Class="ABC123"),
                                ExtraInformation="",
                                Attributes=[]
                            )
                        ],
                        Result="",
                        ExtraVisionResult="",
                        TriggeredGroup="",
                        UserInfo=UserInfoModel()
                    )
                ],
                PipelineOption="Test Pipeline"
            )
        ],
        Images=[
            ImageInfoModel(
                ID=1,
                Image=TEST_IMAGES[1],
                ScaledFactor=1.0
            )
        ]
    )


async def test_queue_system():
    """Test the queue system functionality."""
    # Load configuration
    config = load_config()
    
    # Create image processor
    image_processor = ImageProcessor(
        cache_dir=config.image_processor.cache_dir,
        max_cache_size_mb=config.image_processor.max_cache_size_mb
    )
    
    # Create LLM analyzer if available in config
    llm_analyzer = None
    if hasattr(config, 'llm'):
        # Create LLM analyzer with the config
        try:
            llm_analyzer = LLMAnalyzer(config=config.llm)
            logger.info(f"Created LLM analyzer with provider: {config.llm.provider}")
        except Exception as e:
            logger.error(f"Failed to create LLM analyzer: {e}")
    
    # Create queue system
    queue_system = QueueSystem(
        image_processor=image_processor,
        llm_analyzer=llm_analyzer,
        ui_update_callback=handle_ui_update
    )
    
    try:
        # Start queue system
        await queue_system.start()
        logger.info("Queue system started")
        
        # Create sample data
        detection_data = create_sample_detection_data()
        alpr_event_data = create_sample_alpr_event_data()
        
        # Test with detection data
        logger.info("Testing with DetectionData...")
        await queue_system.put_mqtt_message(detection_data)
        
        # Wait a bit for processing
        await asyncio.sleep(5)
        
        # Test with ALPR event data
        logger.info("Testing with ALPREventData...")
        await queue_system.put_mqtt_message(alpr_event_data)
        
        # Wait for processing to complete
        await asyncio.sleep(10)
        
        # Log queue system statistics
        stats = queue_system.get_stats()
        logger.info("Queue System Statistics:")
        logger.info(f"  MQTT messages received: {stats['mqtt_messages_received']}")
        logger.info(f"  Images processed: {stats['images_processed']}")
        logger.info(f"  LLM analyses performed: {stats['llm_analyses_performed']}")
        logger.info(f"  UI updates sent: {stats['ui_updates_sent']}")
        logger.info(f"  Errors: {stats['errors']}")
        
        # Log queue sizes
        queue_sizes = stats['queue_sizes']
        logger.info("Queue Sizes:")
        logger.info(f"  MQTT: {queue_sizes['mqtt']}")
        logger.info(f"  Image: {queue_sizes['image']}")
        logger.info(f"  LLM: {queue_sizes['llm']}")
        logger.info(f"  UI: {queue_sizes['ui']}")
        
        # Log average processing times
        logger.info("Average Processing Times (ms):")
        logger.info(f"  MQTT: {stats.get('avg_mqtt_time_ms', 0):.2f}")
        logger.info(f"  Image: {stats.get('avg_image_time_ms', 0):.2f}")
        logger.info(f"  LLM: {stats.get('avg_llm_time_ms', 0):.2f}")
        logger.info(f"  UI: {stats.get('avg_ui_time_ms', 0):.2f}")
        
    finally:
        # Stop queue system
        await queue_system.stop()
        logger.info("Queue system stopped")
        
        # Close resources
        await image_processor.close()
        if llm_analyzer:
            await llm_analyzer.close()


async def test_main():
    """Main test function."""
    try:
        await test_queue_system()
    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True)


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_main())
