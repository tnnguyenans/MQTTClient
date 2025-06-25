# MQTT Client with LLM Image Analysis

## Project Overview
A Python-based MQTT client application that subscribes to a Mosquitto MQTT broker to receive object detection data in JSON format. The application uses Large Language Models (LLM) to analyze images and extract additional attributes (color, actions, etc.) beyond basic object detection. Features a real-time web UI using producer/consumer pattern for live data updates and image display.

## Technology Stack
- **Core Language**: Python 3.9+
- **Package Manager**: uv (for project initialization and dependency management)
- **MQTT Client**: paho-mqtt
- **Web Framework**: FastAPI + WebSockets
- **Frontend**: HTML/CSS/JavaScript with WebSocket support
- **LLM Integration**: OpenAI API or Ollama (local LLM)
- **Image Processing**: Pillow (PIL), requests
- **Data Models**: Pydantic
- **Async Processing**: asyncio, asyncio.Queue
- **Database**: SQLite with SQLModel/SQLAlchemy (for detection history)
- **Testing**: pytest, pytest-asyncio
- **UI Serving**: Uvicorn (ASGI server)

## Features (Build in Order)

### Feature 1: Project Setup and MQTT Client
- **Description**: Initialize project structure with uv, create basic MQTT client that connects to Mosquitto broker and subscribes to specified topic
- **Technology**: uv, paho-mqtt, pydantic
- **Action**: Connect to MQTT broker, subscribe to topic, receive and parse JSON messages
- **Steps**: 
  1. Initialize project with `uv init`
  2. Add MQTT dependencies
  3. Create configuration management
  4. Implement basic MQTT subscriber
  5. Add JSON validation with Pydantic models
- **Status**: [ ] Not Started

### Feature 2: Data Models and Validation
- **Description**: Define Pydantic models for object detection data, including objects, bounding boxes, confidence scores, and image links
- **Technology**: Pydantic, typing
- **Action**: Validate incoming JSON data structure and ensure data integrity
- **Steps**:
  1. Create detection data model (objects, bounding boxes, confidence, image_url)
  2. Create validation schemas
  3. Add error handling for malformed data
  4. Create data transformation utilities
- **Status**: [ ] Not Started

### Feature 3: Image Download and Processing
- **Description**: Download images from URLs provided in detection data, implement caching and basic image processing
- **Technology**: requests, Pillow (PIL), aiofiles, asyncio
- **Action**: Asynchronously download images, cache locally, prepare for LLM analysis
- **Steps**:
  1. Implement async image downloader
  2. Add local image caching system
  3. Create image preprocessing utilities
  4. Add image validation and error handling
- **Status**: [ ] Not Started

### Feature 4: LLM Integration for Image Analysis
- **Description**: Integrate LLM (OpenAI API or local Ollama) to analyze images and extract additional attributes like colors, actions, descriptions
- **Technology**: OpenAI API or Ollama, base64 encoding, asyncio
- **Action**: Send images to LLM with detection context, receive enhanced attribute analysis
- **Steps**:
  1. Implement LLM client (OpenAI or Ollama)
  2. Create image-to-text analysis prompts
  3. Add attribute extraction logic (colors, actions, descriptions)
  4. Implement async LLM processing queue
  5. Add LLM response parsing and validation
- **Status**: [ ] Not Started

### Feature 5: Producer/Consumer Pattern Implementation
- **Description**: Implement producer/consumer pattern using asyncio queues for handling MQTT messages, image processing, and LLM analysis
- **Technology**: asyncio, asyncio.Queue, threading
- **Action**: Separate data ingestion, processing, and UI updates into independent async workers
- **Steps**:
  1. Create MQTT message producer
  2. Implement image processing consumer
  3. Create LLM analysis consumer
  4. Add UI update consumer
  5. Implement queue management and flow control
- **Status**: [ ] Not Started

### Feature 6: Database Layer for Detection History
- **Description**: Store detection history, processed images, and LLM analysis results in SQLite database
- **Technology**: SQLModel, SQLAlchemy, SQLite, alembic
- **Action**: Persist detection data, enable historical queries and analytics
- **Steps**:
  1. Create database models (Detection, Image, Analysis)
  2. Implement database connection and session management
  3. Add CRUD operations
  4. Create database migration system
  5. Add data cleanup and archiving
- **Status**: [ ] Not Started

### Feature 7: FastAPI Backend with WebSocket Support
- **Description**: Create FastAPI backend with REST API endpoints and WebSocket connections for real-time UI updates
- **Technology**: FastAPI, WebSockets, uvicorn
- **Action**: Serve API endpoints for detection data and establish WebSocket connections for live updates
- **Steps**:
  1. Create FastAPI application structure
  2. Implement REST API endpoints (GET detections, images, analysis)
  3. Add WebSocket endpoint for real-time updates
  4. Implement WebSocket connection management
  5. Create API response models
- **Status**: [ ] Not Started

### Feature 8: Web UI for Desktop Browser
- **Description**: Create responsive web interface displaying real-time detection data, images, and LLM analysis results
- **Technology**: HTML5, CSS3, JavaScript, WebSocket API, Chart.js (for analytics)
- **Action**: Display live detection feed, image gallery, attribute analysis, and basic analytics dashboard
- **Steps**:
  1. Create HTML templates and CSS styling
  2. Implement JavaScript WebSocket client
  3. Add real-time detection display components
  4. Create image viewer with bounding box overlay
  5. Add LLM analysis results display
  6. Implement basic analytics dashboard
- **Status**: [ ] Not Started

### Feature 9: Configuration and Deployment
- **Description**: Add comprehensive configuration management, logging, and deployment setup
- **Technology**: pydantic-settings, python-dotenv, logging, docker (optional)
- **Action**: Configure MQTT settings, LLM settings, database settings, and deployment options
- **Steps**:
  1. Create configuration models with environment variable support
  2. Add comprehensive logging system
  3. Create startup/shutdown procedures
  4. Add health check endpoints
  5. Create deployment documentation
  6. Optional: Add Docker containerization
- **Status**: [ ] Not Started

### Feature 10: Testing and Quality Assurance
- **Description**: Comprehensive test suite covering all components with unit tests, integration tests, and end-to-end tests
- **Technology**: pytest, pytest-asyncio, pytest-mock, httpx
- **Action**: Ensure application reliability and maintainability through thorough testing
- **Steps**:
  1. Create unit tests for all modules
  2. Add integration tests for MQTT, LLM, and database components
  3. Implement API endpoint tests
  4. Add WebSocket connection tests
  5. Create performance and load tests
  6. Add test data fixtures and mocks
- **Status**: [ ] Not Started

## Project Structure
```
mqtt_client/
├── src/
│   ├── mqtt_client/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── detection.py
│   │   │   └── analysis.py
│   │   ├── mqtt/
│   │   │   ├── __init__.py
│   │   │   └── client.py
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   └── analyzer.py
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   └── models.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py
│   │   ├── processors/
│   │   │   ├── __init__.py
│   │   │   ├── image.py
│   │   │   └── queue_manager.py
│   │   └── static/
│   │       ├── index.html
│   │       ├── style.css
│   │       └── app.js
├── tests/
│   ├── __init__.py
│   ├── test_mqtt.py
│   ├── test_llm.py
│   ├── test_api.py
│   └── test_processors.py
├── pyproject.toml
├── README.md
├── .env.example
└── INSTRUCTION.md
```

## Development Notes
- Use async/await patterns throughout for better performance
- Implement proper error handling and retry mechanisms for MQTT and LLM calls
- Add rate limiting for LLM API calls to manage costs
- Consider image size optimization before sending to LLM
- Implement graceful shutdown procedures for all async components
- Add monitoring and metrics collection for production deployment
