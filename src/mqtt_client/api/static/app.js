/**
 * MQTT Client Dashboard
 * Handles WebSocket communication and UI interactions for real-time detection data
 */

// Store global state
const state = {
    websocket: null,
    connected: false,
    detections: [],
    selectedDetection: null,
    maxDetections: 100, // Maximum number of detections to keep in memory
};

// Global elements object to store DOM references
const elements = {};

/**
 * Shows a placeholder message when an image fails to load
 * @param {string} message - Message to display in the placeholder
 */
function showPlaceholderImage(message) {
    // Clear any existing content
    elements.imageViewer.innerHTML = '';
    
    // Create placeholder element
    const placeholder = document.createElement('div');
    placeholder.className = 'placeholder-image';
    placeholder.style.padding = '20px';
    placeholder.style.backgroundColor = '#f8f9fa';
    placeholder.style.border = '1px solid #dee2e6';
    placeholder.style.borderRadius = '4px';
    placeholder.style.textAlign = 'center';
    placeholder.style.color = '#6c757d';
    placeholder.textContent = message || 'No image available';
    
    // Add to DOM
    elements.imageViewer.appendChild(placeholder);
}

// DOM elements cache
elements.statusIndicator = null;
elements.statusText = null;
elements.eventList = null;
elements.imageViewer = null;
elements.detectionInfo = null;
elements.analysisSection = null;

// Initialize the application when DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    // Cache DOM elements
    cacheElements();
    
    // Initialize WebSocket connection
    initWebSocket();
    
    // Load initial detection history
    loadDetectionHistory();
    
    // Add event listeners
    document.getElementById('refresh-btn').addEventListener('click', loadDetectionHistory);
});

// Cache frequently used DOM elements for better performance
function cacheElements() {
    elements.statusIndicator = document.querySelector('.status-indicator');
    elements.statusText = document.querySelector('.status-text');
    elements.eventList = document.getElementById('event-list');
    elements.imageViewer = document.getElementById('image-viewer');
    elements.detectionInfo = document.getElementById('detection-info');
    elements.analysisSection = document.getElementById('analysis-section');
}

// Initialize WebSocket connection
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    // Use explicit port 8089 to match server configuration
    const host = window.location.hostname;
    const port = 8089;
    const wsUrl = `${protocol}${host}:${port}/ws`;
    
    console.log(`Attempting to connect to WebSocket at: ${wsUrl}`);
    updateConnectionStatus('connecting');
    
    state.websocket = new WebSocket(wsUrl);
    
    // WebSocket event handlers
    state.websocket.onopen = handleWebSocketOpen;
    state.websocket.onmessage = handleWebSocketMessage;
    state.websocket.onclose = handleWebSocketClose;
    state.websocket.onerror = handleWebSocketError;
    
    // Set up automatic reconnection
    window.addEventListener('online', tryReconnect);
}

// Handle WebSocket open event
function handleWebSocketOpen() {
    console.log('WebSocket connection established');
    updateConnectionStatus('online');
}

// Handle incoming WebSocket messages
function handleWebSocketMessage(event) {
    try {
        const data = JSON.parse(event.data);
        console.log('Received WebSocket message:', data);
        
        if (data.type === 'detection') {
            handleNewDetection(data.data);
        } else if (data.type === 'analysis') {
            updateAnalysis(data.data);
        } else if (data.type === 'system') {
            // Handle system messages if needed
            console.log('System message:', data.message);
        }
    } catch (error) {
        console.error('Error processing WebSocket message:', error);
    }
}

// Handle WebSocket close event
function handleWebSocketClose() {
    console.log('WebSocket connection closed');
    updateConnectionStatus('offline');
    
    // Try to reconnect after a delay
    setTimeout(tryReconnect, 5000);
}

// Handle WebSocket error
function handleWebSocketError(error) {
    console.error('WebSocket error:', error);
    console.error('WebSocket error details:', JSON.stringify(error));
    updateConnectionStatus('offline');
}

// Try to reestablish WebSocket connection
function tryReconnect() {
    if (!state.connected && navigator.onLine) {
        console.log('Attempting to reconnect WebSocket...');
        initWebSocket();
    }
}

// Update connection status UI
function updateConnectionStatus(status) {
    state.connected = status === 'online';
    
    // Update the status indicator class
    elements.statusIndicator.className = 'status-indicator';
    elements.statusIndicator.classList.add(`status-${status}`);
    
    // Update status text
    switch (status) {
        case 'online':
            elements.statusText.textContent = 'Connected';
            break;
        case 'offline':
            elements.statusText.textContent = 'Disconnected';
            break;
        case 'connecting':
            elements.statusText.textContent = 'Connecting...';
            break;
    }
}

// Load detection history from the API
async function loadDetectionHistory() {
    try {
        console.log('Loading detection history...');
        // Use absolute paths for API endpoints
        const response = await fetch('/detections?limit=50');
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Received detection history:', data);
        
        // Check if data is an array directly or has an items property
        if (Array.isArray(data)) {
            state.detections = data;
        } else if (data.items && Array.isArray(data.items)) {
            state.detections = data.items;
        } else {
            state.detections = [];
            console.warn('Unexpected data format for detections:', data);
        }
        
        console.log(`Loaded ${state.detections.length} detections`);
        
        // Update the UI with the detection history
        renderDetectionList();
        
        // Select the first detection if available and none is selected
        if (state.detections.length > 0 && !state.selectedDetection) {
            selectDetection(state.detections[0]);
        }
    } catch (error) {
        console.error('Error loading detection history:', error);
        showError('Failed to load detection history');
    }
}

// Handle new detection from WebSocket
function handleNewDetection(detection) {
    // Add to the beginning of the array (newest first)
    state.detections.unshift(detection);
    
    // Limit the number of detections kept in memory
    if (state.detections.length > state.maxDetections) {
        state.detections = state.detections.slice(0, state.maxDetections);
    }
    
    // Update the UI
    renderDetectionList();
    
    // Auto-select the new detection if enabled
    const autoSelectNew = true; // Could be a user preference setting
    if (autoSelectNew) {
        selectDetection(detection);
    }
}

// Render the detection list in the sidebar
function renderDetectionList() {
    // Clear the event list
    elements.eventList.innerHTML = '';
    
    // If no detections, show a message
    if (state.detections.length === 0) {
        const noData = document.createElement('div');
        noData.className = 'no-data';
        noData.textContent = 'No detection events available';
        elements.eventList.appendChild(noData);
        return;
    }
    
    // Create an element for each detection
    state.detections.forEach(detection => {
        const eventItem = document.createElement('div');
        eventItem.className = 'event-item';
        if (state.selectedDetection && detection.id === state.selectedDetection.id) {
            eventItem.classList.add('selected');
        }
        
        // Format the detection information
        const detectionTime = new Date(detection.timestamp);
        const detectionName = detection.camera_name || detection.source || 'Unknown';
        const detectionObjects = Array.isArray(detection.objects) ? detection.objects.length : 0;
        
        // Create detection item content
        eventItem.innerHTML = `
            <div>${detectionName}</div>
            <div class="event-time">${formatDateTime(detectionTime)} · ${detectionObjects} object(s)</div>
        `;
        
        // Add click event to select this detection
        eventItem.addEventListener('click', () => {
            selectDetection(detection);
        });
        
        elements.eventList.appendChild(eventItem);
    });
}

// Select a detection and update the UI
function selectDetection(detection) {
    state.selectedDetection = detection;
    
    // Update the selection in the list
    const items = elements.eventList.querySelectorAll('.event-item');
    items.forEach(item => {
        item.classList.remove('selected');
        // Find the item that matches the selected detection and highlight it
        const itemTime = item.querySelector('.event-time').textContent;
        if (itemTime.includes(formatDateTime(new Date(detection.timestamp)))) {
            item.classList.add('selected');
        }
    });
    
    // Update the image viewer
    updateImageViewer(detection);
    
    // Update detection information
    updateDetectionInfo(detection);
    
    // Update analysis section if available
    if (detection.analysis) {
        updateAnalysis(detection.analysis);
    } else {
        // Clear analysis section if no analysis available
        elements.analysisSection.innerHTML = '<p class="no-data">No analysis available for this detection</p>';
    }
}

// Update image viewer with detection image
function updateImageViewer(detection) {
    // Clear current content
    elements.imageViewer.innerHTML = '';
    
    // Get image URL if available
    const imageUrl = detection.image_url;
    
    if (imageUrl) {
        const img = document.createElement('img');
        
        // Helper function to check if a string might be base64 encoded
        function isLikelyBase64(str) {
            // Check if the string is very long and contains base64 characters
            if (str.length > 100) {
                // Base64 uses these characters: A-Z, a-z, 0-9, +, /, and = for padding
                const base64Pattern = /^[A-Za-z0-9+/=]+$/;
                // Check if at least 90% of characters match base64 pattern
                const validChars = str.split('').filter(char => base64Pattern.test(char)).length;
                return validChars / str.length > 0.9;
            }
            return false;
        }
        
        // Handle different image URL formats
        if (imageUrl.startsWith('data:')) {
            // For base64 images with proper data URI format, use directly
            img.src = imageUrl;
        } else if (imageUrl.startsWith('http')) {
            // For external URLs, try direct access first
            const filename = imageUrl.split('/').pop();
            img.src = imageUrl;
            
            // If direct access fails, fall back to cached version
            img.onerror = function() {
                console.log('Direct URL access failed, trying cached version');
                img.src = `/images/${filename}`;
                
                // If cached version fails, show placeholder
                img.onerror = function() {
                    console.error('Failed to load image from cache:', filename);
                    showPlaceholderImage('Image failed to load');
                };
            };
        } else if (isLikelyBase64(imageUrl)) {
            // If it looks like a base64 string without the data URI prefix, use our image endpoint
            console.log('Detected likely raw image data, using image endpoint');
            
            // Create a unique blob URL for this image as a placeholder while loading
            const blobUrl = URL.createObjectURL(new Blob(['loading'], {type: 'text/plain'}));
            img.src = blobUrl;
            
            // Add loading indicator
            const loadingIndicator = document.createElement('div');
            loadingIndicator.className = 'loading-indicator';
            loadingIndicator.textContent = 'Loading image...';
            elements.imageViewer.appendChild(loadingIndicator);
            
            // Make a POST request to the image endpoint
            fetch('/api/image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ data: imageUrl })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Image endpoint returned status ' + response.status);
                }
                return response.blob();
            })
            .then(blob => {
                // Check if the blob is valid (non-empty)
                if (blob.size === 0) {
                    throw new Error('Empty image data received');
                }
                
                // Create a URL for the blob and set it as the image source
                const imageUrl = URL.createObjectURL(blob);
                img.src = imageUrl;
                img.style.width = '100%';
                img.style.height = 'auto';
                img.onload = function() {
                    // Ensure the image container expands to fit the image
                    const imageViewer = document.getElementById('image-viewer');
                    imageViewer.style.height = 'auto';
                };
                console.log('Successfully loaded image from endpoint');
                
                // Remove loading indicator
                if (loadingIndicator.parentNode) {
                    loadingIndicator.parentNode.removeChild(loadingIndicator);
                }
            })
            .catch(error => {
                console.error('Error loading image:', error);
                
                // Try with data URI as fallback
                console.log('Image endpoint failed, trying with data URI prefix');
                
                // First try with direct base64 encoding
                img.src = `data:image/jpeg;base64,${imageUrl}`;
                
                // If that fails, try with cleaned base64 string
                img.onerror = function() {
                    console.log('Direct data URI failed, trying with cleaned base64');
                    // Clean the string to only include valid base64 characters
                    const cleanedData = imageUrl.replace(/[^A-Za-z0-9+/=]/g, '');
                    img.src = `data:image/jpeg;base64,${cleanedData}`;
                    
                    // If that also fails, show placeholder
                    img.onerror = function() {
                        console.error('All attempts to load image failed');
                        showPlaceholderImage('Image failed to load');
                    };
                };
                
                // Remove loading indicator
                if (loadingIndicator.parentNode) {
                    loadingIndicator.parentNode.removeChild(loadingIndicator);
                }
            });
        } else {
            // For relative paths or just filenames
            img.src = `/images/${imageUrl}`;
        }
        
        img.alt = 'Detection Image';
        img.onload = function() {
            // After image loads, add bounding boxes if available
            if (detection.objects && detection.objects.length > 0) {
                addBoundingBoxes(detection.objects, img);
            }
        };
        img.onerror = function() {
            console.error('Failed to load image:', img.src);
            // Show placeholder if image fails to load
            showPlaceholderImage('Image failed to load');
        };
        

        elements.imageViewer.appendChild(img);
    } else {
        showPlaceholderImage('No image available');
    }
}

// Add bounding boxes overlay on the image
function addBoundingBoxes(objects, img) {
    // Calculate scaling factor for bounding boxes
    const imgNaturalWidth = img.naturalWidth;
    const imgNaturalHeight = img.naturalHeight;
    const imgDisplayWidth = img.clientWidth;
    const imgDisplayHeight = img.clientHeight;
    
    const scaleX = imgDisplayWidth / imgNaturalWidth;
    const scaleY = imgDisplayHeight / imgNaturalHeight;
    
    // Create SVG overlay for bounding boxes
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.style.position = 'absolute';
    svg.style.top = '0';
    svg.style.left = '0';
    svg.style.pointerEvents = 'none';
    
    // Add each bounding box
    objects.forEach((obj, index) => {
        if (obj.bounding_box) {
            const { left, top, right, bottom } = obj.bounding_box;
            
            // Calculate scaled coordinates
            const x = left * scaleX;
            const y = top * scaleY;
            const width = (right - left) * scaleX;
            const height = (bottom - top) * scaleY;
            
            // Create rectangle for bounding box
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', x);
            rect.setAttribute('y', y);
            rect.setAttribute('width', width);
            rect.setAttribute('height', height);
            rect.setAttribute('fill', 'none');
            rect.setAttribute('stroke', generateColor(index));
            rect.setAttribute('stroke-width', '2');
            
            // Add label if available
            if (obj.class_name) {
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', x);
                text.setAttribute('y', y - 5);
                text.setAttribute('fill', generateColor(index));
                text.setAttribute('font-size', '12px');
                text.textContent = obj.class_name;
                svg.appendChild(text);
            }
            
            svg.appendChild(rect);
        }
    });
    
    // Add the SVG overlay to the image container
    elements.imageViewer.appendChild(svg);
}

// Show placeholder when image is not available
function showPlaceholderImage(message) {
    const placeholder = document.createElement('div');
    placeholder.className = 'placeholder-image';
    placeholder.textContent = message;
    elements.imageViewer.appendChild(placeholder);
}

// Update detection information panel
function updateDetectionInfo(detection) {
    // Clear current content
    elements.detectionInfo.innerHTML = '';
    
    // Show detection metadata
    const metadataDiv = document.createElement('div');
    metadataDiv.className = 'detection-metadata';
    metadataDiv.innerHTML = `
        <h3>Detection Information</h3>
        <p><strong>Time:</strong> ${formatDateTime(new Date(detection.timestamp))}</p>
        <p><strong>Camera:</strong> ${detection.camera_name || 'Unknown'}</p>
        <p><strong>Source:</strong> ${detection.source || 'Unknown'}</p>
    `;
    elements.detectionInfo.appendChild(metadataDiv);
    
    // Show detected objects
    if (detection.objects && detection.objects.length > 0) {
        const objectsDiv = document.createElement('div');
        objectsDiv.innerHTML = '<h3>Detected Objects</h3>';
        
        detection.objects.forEach(obj => {
            const objectDiv = document.createElement('div');
            objectDiv.className = 'detection-object';
            
            const confidence = obj.confidence ? Math.round(obj.confidence * 100) : 'N/A';
            
            objectDiv.innerHTML = `
                <div class="detection-header">
                    <span>${obj.class_name || 'Unknown Object'}</span>
                    <span class="detection-confidence">${confidence}%</span>
                </div>
            `;
            
            // Add attributes if available
            if (obj.attributes && Object.keys(obj.attributes).length > 0) {
                const attrList = document.createElement('div');
                attrList.className = 'tag-list';
                
                for (const [key, value] of Object.entries(obj.attributes)) {
                    const tag = document.createElement('span');
                    tag.className = 'tag';
                    tag.textContent = `${key}: ${value}`;
                    attrList.appendChild(tag);
                }
                
                objectDiv.appendChild(attrList);
            }
            
            objectsDiv.appendChild(objectDiv);
        });
        
        elements.detectionInfo.appendChild(objectsDiv);
    } else {
        const noObjects = document.createElement('p');
        noObjects.className = 'no-data';
        noObjects.textContent = 'No objects detected';
        elements.detectionInfo.appendChild(noObjects);
    }
}

// Update analysis section with LLM results
function updateAnalysis(analysis) {
    // Clear current content
    elements.analysisSection.innerHTML = '<h3>LLM Analysis</h3>';
    
    // Check if we have valid analysis data
    if (!analysis || Object.keys(analysis).length === 0) {
        const noData = document.createElement('p');
        noData.className = 'no-data';
        noData.textContent = 'No analysis data available';
        elements.analysisSection.appendChild(noData);
        return;
    }
    
    // Create analysis items
    for (const [key, value] of Object.entries(analysis)) {
        // Skip empty values
        if (!value || (Array.isArray(value) && value.length === 0)) {
            continue;
        }
        
        const analysisItem = document.createElement('div');
        analysisItem.className = 'analysis-item';
        
        // Format the key as a readable title
        const title = key
            .replace(/_/g, ' ')
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
        
        // Create header
        const header = document.createElement('div');
        header.className = 'analysis-header';
        header.textContent = title;
        analysisItem.appendChild(header);
        
        // Create content based on type
        const content = document.createElement('div');
        content.className = 'analysis-content';
        
        // Handle different data types
        if (Array.isArray(value)) {
            const tagList = document.createElement('div');
            tagList.className = 'tag-list';
            
            value.forEach(item => {
                const tag = document.createElement('span');
                tag.className = 'tag';
                tag.textContent = item;
                tagList.appendChild(tag);
            });
            
            content.appendChild(tagList);
        } else if (typeof value === 'object') {
            // For nested objects
            const pre = document.createElement('pre');
            pre.textContent = JSON.stringify(value, null, 2);
            content.appendChild(pre);
        } else {
            // For simple string/number values
            content.textContent = value;
        }
        
        analysisItem.appendChild(content);
        elements.analysisSection.appendChild(analysisItem);
    }
}

// Helper function to format date/time
function formatDateTime(date) {
    return new Intl.DateTimeFormat('default', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
        month: 'short',
        day: 'numeric'
    }).format(date);
}

// Generate a color based on index for consistent object colors
function generateColor(index) {
    const colors = [
        '#3498db', // blue
        '#2ecc71', // green
        '#e74c3c', // red
        '#f39c12', // orange
        '#9b59b6', // purple
        '#1abc9c', // teal
        '#d35400', // dark orange
        '#27ae60', // dark green
        '#c0392b', // dark red
        '#8e44ad'  // dark purple
    ];
    
    return colors[index % colors.length];
}

// Display error message
function showError(message) {
    console.error(message);
    // Could implement a UI toast/notification here
}