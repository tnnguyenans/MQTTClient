# MQTT Client with LLM Image Analysis

A Python-based MQTT client application that subscribes to a Mosquitto MQTT broker to receive object detection data from ANSVIS Server in JSON format. The application uses Large Language Models (LLM) to analyze images and extract additional attributes (color, actions, etc.) beyond basic object detection. Features a real-time web UI using producer/consumer pattern for live data updates and image display.

## Features

- [x] **Project Setup and MQTT Client**: Basic MQTT client that connects to Mosquitto broker and subscribes to specified topic
  - [x] **Graceful Shutdown**: Fixed issues with client exit to ensure proper resource cleanup
  - [x] **Robust Message Parsing**: Enhanced error handling for ALPR bounding box data with case-insensitive attribute access
- [x] **Data Models and Validation**: Pydantic models for object detection data
  - [x] **Flexible Data Models**: Updated BoundingBoxModel to support both uppercase and lowercase attribute names
- [x] **Image Download and Processing**: Download and process images from URLs and base64 strings with caching
- [x] **LLM Integration**: Analyze images using LLM to extract additional attributes (colors, actions, descriptions)
- [x] **Producer/Consumer Pattern**: Implement async processing queue system with MQTT, image, LLM, and UI stages
  - [x] **Improved Concurrency**: Enhanced async task management for reliable operation
- [ ] **FastAPI Backend**: Create API endpoints and WebSocket connections
- [ ] **Web UI**: Display real-time detection data and images
- [ ] **Database Layer**: Store detection history in SQLite
- [ ] **Configuration and Deployment**: Add comprehensive configuration management

## Setup

### Install MQTT Broker

1. Download Mosquitto MQTT broker for Windows: [mosquitto-2.0.21a-install-windows-x64.exe](https://mosquitto.org/files/binary/win64/mosquitto-2.0.21a-install-windows-x64.exe)
2. Run the installer and follow the installation wizard
3. After installation, start the Mosquitto service:
   - Open Command Prompt as Administrator
   - Run: `net start mosquitto`
   - Alternatively, you can set it to start automatically through Windows Services

### Configure Mosquitto

The default configuration file is located at `C:\Program Files\mosquitto\mosquitto.conf`. For development purposes, you may want to enable anonymous access:

```
listener 1883
allow_anonymous true
```

Restart the service after configuration changes:
```
net stop mosquitto
net start mosquitto
```
- [ ] **Testing and Quality Assurance**: Comprehensive test suite

## Installation

### Prerequisites

- Python 3.9 or higher
- uv package manager

### Setup

1. Clone the repository

```bash
git clone <repository-url>
cd mqtt-client
```

2. Install dependencies

```bash
uv install
```

## Usage

### Configure ANSVIS Client with MQTT Trigger
1. Create ANSVIS AI Task

![image](https://github.com/user-attachments/assets/cf05b905-1075-4328-a5fe-0bfbc2f6ba84)

2. Create Trigger for AI Task
   
   ![image](https://github.com/user-attachments/assets/49755ff5-ca1d-4e5b-9976-a8c870dbbd1a)


### Running the MQTT Client

To run the MQTT client in standalone mode:

```bash
python -m mqtt_client.main
```


### Running the Web Server

To run the web server with the MQTT client integrated:

```bash
.\.venv\Scripts\python.exe -m mqtt_client.server
```

Once the server is running, you can access the web UI at:

```
http://localhost:8089
```

## Configuration

The default configuration connects to a local MQTT broker on port 1883 and subscribes to the `detection/objects` topic. You can modify these settings in `src/mqtt_client/config.py`.

## Development

### Running Tests

```bash
python -m pytest
```

## Project Structure

```
mqtt_client/
├── src/
│   └── mqtt_client/
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── detection.py
│       └── mqtt/
│           ├── __init__.py
│           └── client.py
├── tests/
│   ├── __init__.py
│   └── test_mqtt.py
├── pyproject.toml
├── README.md
└── INSTRUCTION.md
```

## License

MIT
