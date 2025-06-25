/**
 * MQTT Client Analytics Dashboard
 * Handles data processing and visualizations for detection analytics
 */

// Store chart instances for later updates
const charts = {
    timeline: null,
    objects: null,
    sources: null
};

// Store analytics data
const analyticsData = {
    detections: [],
    totalDetections: 0,
    uniqueSources: new Set(),
    objectClasses: {},
    objectsDetected: 0,
    confidenceSum: 0
};

// Store global state
const state = {
    websocket: null,
    connected: false
};

// DOM elements cache
const elements = {
    statusIndicator: null,
    statusText: null
};

// Initialize the analytics dashboard
document.addEventListener('DOMContentLoaded', () => {
    // Cache DOM elements
    elements.statusIndicator = document.querySelector('.status-indicator');
    elements.statusText = document.querySelector('.status-text');
    
    // Initialize WebSocket connection
    initWebSocket();
    
    // Fetch detection data
    fetchDetectionData();
    
    // Set up auto-refresh interval (every 60 seconds)
    setInterval(fetchDetectionData, 60000);
});

// Initialize WebSocket connection
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    const wsUrl = `${protocol}${window.location.host}/ws`;
    
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
            // Update analytics data with new detection
            fetchDetectionData();
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

// Fetch detection data from API
async function fetchDetectionData() {
    try {
        const response = await fetch('/detections?limit=100');
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Process the detection data for analytics
        processAnalyticsData(data.items);
        
        // Update the UI with processed data
        updateStatCards();
        updateCharts();
    } catch (error) {
        console.error('Error fetching detection data:', error);
    }
}

// Process detection data for analytics
function processAnalyticsData(detections) {
    // Reset analytics data
    analyticsData.detections = detections;
    analyticsData.totalDetections = detections.length;
    analyticsData.uniqueSources = new Set();
    analyticsData.objectClasses = {};
    analyticsData.objectsDetected = 0;
    analyticsData.confidenceSum = 0;
    
    // Process each detection
    detections.forEach(detection => {
        // Track unique sources
        const source = detection.camera_name || detection.source || 'Unknown';
        analyticsData.uniqueSources.add(source);
        
        // Process objects
        if (detection.objects && Array.isArray(detection.objects)) {
            analyticsData.objectsDetected += detection.objects.length;
            
            // Process each object
            detection.objects.forEach(obj => {
                // Count object classes
                const className = obj.class_name || 'Unknown';
                if (!analyticsData.objectClasses[className]) {
                    analyticsData.objectClasses[className] = 0;
                }
                analyticsData.objectClasses[className]++;
                
                // Sum confidence for average calculation
                if (obj.confidence) {
                    analyticsData.confidenceSum += obj.confidence;
                }
            });
        }
    });
}

// Update stat cards with processed data
function updateStatCards() {
    document.getElementById('total-detections').textContent = analyticsData.totalDetections;
    document.getElementById('unique-sources').textContent = analyticsData.uniqueSources.size;
    document.getElementById('objects-detected').textContent = analyticsData.objectsDetected;
    
    // Calculate average confidence
    const avgConfidence = analyticsData.objectsDetected > 0
        ? Math.round((analyticsData.confidenceSum / analyticsData.objectsDetected) * 100)
        : 0;
    document.getElementById('avg-confidence').textContent = `${avgConfidence}%`;
}

// Update all charts
function updateCharts() {
    updateTimelineChart();
    updateObjectsChart();
    updateSourcesChart();
}

// Update timeline chart
function updateTimelineChart() {
    // Group detections by hour
    const timeData = {};
    
    analyticsData.detections.forEach(detection => {
        const date = new Date(detection.timestamp);
        const hour = date.toISOString().slice(0, 13); // YYYY-MM-DDTHH format
        
        if (!timeData[hour]) {
            timeData[hour] = 0;
        }
        timeData[hour]++;
    });
    
    // Sort hours chronologically
    const sortedHours = Object.keys(timeData).sort();
    
    // Format labels to be more readable
    const labels = sortedHours.map(hour => {
        const date = new Date(hour);
        return date.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', hour12: false });
    });
    
    // Prepare chart data
    const chartData = {
        labels: labels,
        datasets: [{
            label: 'Detections per Hour',
            data: sortedHours.map(hour => timeData[hour]),
            backgroundColor: 'rgba(52, 152, 219, 0.5)',
            borderColor: 'rgba(52, 152, 219, 1)',
            borderWidth: 1
        }]
    };
    
    // Create or update chart
    const ctx = document.getElementById('timeline-chart').getContext('2d');
    
    if (charts.timeline) {
        charts.timeline.data = chartData;
        charts.timeline.update();
    } else {
        charts.timeline = new Chart(ctx, {
            type: 'bar',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Detections'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    }
                }
            }
        });
    }
}

// Update objects distribution chart
function updateObjectsChart() {
    // Get top 10 object classes by count
    const objectEntries = Object.entries(analyticsData.objectClasses)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10); // Top 10
    
    const labels = objectEntries.map(entry => entry[0]);
    const data = objectEntries.map(entry => entry[1]);
    
    // Generate colors
    const backgroundColor = labels.map((_, index) => generateChartColor(index, 0.7));
    const borderColor = labels.map((_, index) => generateChartColor(index, 1));
    
    // Prepare chart data
    const chartData = {
        labels: labels,
        datasets: [{
            label: 'Object Classes',
            data: data,
            backgroundColor: backgroundColor,
            borderColor: borderColor,
            borderWidth: 1
        }]
    };
    
    // Create or update chart
    const ctx = document.getElementById('objects-chart').getContext('2d');
    
    if (charts.objects) {
        charts.objects.data = chartData;
        charts.objects.update();
    } else {
        charts.objects = new Chart(ctx, {
            type: 'pie',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;
                                const percentage = Math.round((value / analyticsData.objectsDetected) * 100);
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }
}

// Update sources distribution chart
function updateSourcesChart() {
    // Count detections by source
    const sourceData = {};
    
    analyticsData.detections.forEach(detection => {
        const source = detection.camera_name || detection.source || 'Unknown';
        
        if (!sourceData[source]) {
            sourceData[source] = 0;
        }
        sourceData[source]++;
    });
    
    // Sort sources by count
    const sourceEntries = Object.entries(sourceData)
        .sort((a, b) => b[1] - a[1]);
    
    const labels = sourceEntries.map(entry => entry[0]);
    const data = sourceEntries.map(entry => entry[1]);
    
    // Generate colors
    const backgroundColor = labels.map((_, index) => generateChartColor(index, 0.7));
    const borderColor = labels.map((_, index) => generateChartColor(index, 1));
    
    // Prepare chart data
    const chartData = {
        labels: labels,
        datasets: [{
            label: 'Detections by Source',
            data: data,
            backgroundColor: backgroundColor,
            borderColor: borderColor,
            borderWidth: 1
        }]
    };
    
    // Create or update chart
    const ctx = document.getElementById('sources-chart').getContext('2d');
    
    if (charts.sources) {
        charts.sources.data = chartData;
        charts.sources.update();
    } else {
        charts.sources = new Chart(ctx, {
            type: 'doughnut',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;
                                const percentage = Math.round((value / analyticsData.totalDetections) * 100);
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }
}

// Generate colors for charts based on index
function generateChartColor(index, alpha = 1) {
    const colors = [
        `rgba(52, 152, 219, ${alpha})`,  // blue
        `rgba(46, 204, 113, ${alpha})`,  // green
        `rgba(231, 76, 60, ${alpha})`,   // red
        `rgba(243, 156, 18, ${alpha})`,  // orange
        `rgba(155, 89, 182, ${alpha})`,  // purple
        `rgba(26, 188, 156, ${alpha})`,  // teal
        `rgba(211, 84, 0, ${alpha})`,    // dark orange
        `rgba(39, 174, 96, ${alpha})`,   // dark green
        `rgba(192, 57, 43, ${alpha})`,   // dark red
        `rgba(142, 68, 173, ${alpha})`   // dark purple
    ];
    
    return colors[index % colors.length];
}