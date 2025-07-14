"""
Test script for multi-column and multi-axis chart rendering in the Conversational Query Engine.
This script creates a local HTTP server to serve a test page that renders various chart types
with multi-column data and secondary y-axis support.
"""

import os
import json
import webbrowser
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs
import threading
import time

# Sample data for testing
SAMPLE_DATA = [
    {"QUARTER": "Q1", "TOTAL_SALES": 123047757, "TOTAL_QUANTITY_SOLD": 2148484, "AVG_PRICE": 57.27},
    {"QUARTER": "Q2", "TOTAL_SALES": 150031127, "TOTAL_QUANTITY_SOLD": 2404085, "AVG_PRICE": 62.41},
    {"QUARTER": "Q3", "TOTAL_SALES": 158468348, "TOTAL_QUANTITY_SOLD": 2427169, "AVG_PRICE": 65.29},
    {"QUARTER": "Q4", "TOTAL_SALES": 144055215, "TOTAL_QUANTITY_SOLD": 2436600, "AVG_PRICE": 59.12}
]

# Sample chart recommendations
CHART_RECOMMENDATIONS = [
    {
        "chart_type": "line",
        "reasoning": "Line chart shows trends over time periods",
        "priority": 1,
        "chart_config": {
            "title": "Quarterly Sales and Quantity Trends",
            "x_axis": "QUARTER",
            "y_axis": ["TOTAL_SALES", "TOTAL_QUANTITY_SOLD"],
            "chart_library": "plotly",
            "additional_config": {
                "use_secondary_axis": True,
                "secondary_axis_columns": ["TOTAL_QUANTITY_SOLD"]
            }
        }
    },
    {
        "chart_type": "bar",
        "reasoning": "Bar chart compares values across categories",
        "priority": 2,
        "chart_config": {
            "title": "Quarterly Sales and Average Price",
            "x_axis": "QUARTER",
            "y_axis": ["TOTAL_SALES", "AVG_PRICE"],
            "chart_library": "plotly",
            "additional_config": {
                "use_secondary_axis": True,
                "secondary_axis_columns": ["AVG_PRICE"]
            }
        }
    },
    {
        "chart_type": "mixed",
        "reasoning": "Mixed chart shows different metrics with appropriate visualizations",
        "priority": 3,
        "chart_config": {
            "title": "Quarterly Performance Mixed Chart",
            "x_axis": "QUARTER",
            "series": [
                {
                    "column": "TOTAL_SALES",
                    "chart_type": "bar",
                    "axis": "primary"
                },
                {
                    "column": "TOTAL_QUANTITY_SOLD",
                    "chart_type": "line",
                    "axis": "secondary"
                },
                {
                    "column": "AVG_PRICE",
                    "chart_type": "scatter",
                    "axis": "secondary"
                }
            ],
            "chart_library": "plotly",
            "additional_config": {
                "use_secondary_axis": True
            }
        }
    }
]

# HTML template for the test page
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Axis Chart Test</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #333;
        }
        .chart-container {
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .chart {
            width: 100%;
            height: 400px;
        }
        .debug-info {
            margin-top: 10px;
            padding: 10px;
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
        }
        pre {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        .nav-tabs {
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Multi-Axis Chart Test</h1>
        <p>This page tests the enhanced chart rendering capabilities with multi-column and multi-axis support.</p>
        
        <div class="chart-container">
            <h2>Test Data</h2>
            <pre id="test-data">SAMPLE_DATA_PLACEHOLDER</pre>
        </div>
        
        <ul class="nav nav-tabs" id="chartTabs" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="chart1-tab" data-bs-toggle="tab" data-bs-target="#chart1" type="button" role="tab">Line Chart</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="chart2-tab" data-bs-toggle="tab" data-bs-target="#chart2" type="button" role="tab">Bar Chart</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="chart3-tab" data-bs-toggle="tab" data-bs-target="#chart3" type="button" role="tab">Mixed Chart</button>
            </li>
        </ul>
        
        <div class="tab-content" id="chartTabsContent">
            <div class="tab-pane fade show active" id="chart1" role="tabpanel">
                <div class="chart-container">
                    <h3>Multi-Column Line Chart with Secondary Y-Axis</h3>
                    <div id="line-chart" class="chart"></div>
                    <div class="debug-info" id="line-chart-debug"></div>
                </div>
            </div>
            <div class="tab-pane fade" id="chart2" role="tabpanel">
                <div class="chart-container">
                    <h3>Multi-Column Bar Chart with Secondary Y-Axis</h3>
                    <div id="bar-chart" class="chart"></div>
                    <div class="debug-info" id="bar-chart-debug"></div>
                </div>
            </div>
            <div class="tab-pane fade" id="chart3" role="tabpanel">
                <div class="chart-container">
                    <h3>Mixed Chart (Bar + Line + Scatter)</h3>
                    <div id="mixed-chart" class="chart"></div>
                    <div class="debug-info" id="mixed-chart-debug"></div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Sample data and chart recommendations
        const testData = SAMPLE_DATA_PLACEHOLDER;
        const chartRecommendations = CHART_RECOMMENDATIONS_PLACEHOLDER;
        
        // Display the test data
        document.getElementById('test-data').textContent = JSON.stringify(testData, null, 2);
        
        // Load chart rendering functions from chart_viewer.html
        CHART_FUNCTIONS_PLACEHOLDER
        
        // Render the charts when the page loads
        document.addEventListener('DOMContentLoaded', function() {
            try {
                // Render line chart
                const lineChartConfig = chartRecommendations[0].chart_config;
                renderChart(lineChartConfig, testData, 'line-chart');
                document.getElementById('line-chart-debug').textContent = 
                    `Chart Type: ${chartRecommendations[0].chart_type}\n` +
                    `Config: ${JSON.stringify(lineChartConfig, null, 2)}`;
                
                // Render bar chart
                const barChartConfig = chartRecommendations[1].chart_config;
                renderChart(barChartConfig, testData, 'bar-chart');
                document.getElementById('bar-chart-debug').textContent = 
                    `Chart Type: ${chartRecommendations[1].chart_type}\n` +
                    `Config: ${JSON.stringify(barChartConfig, null, 2)}`;
                
                // Render mixed chart
                const mixedChartConfig = chartRecommendations[2].chart_config;
                renderChart(mixedChartConfig, testData, 'mixed-chart');
                document.getElementById('mixed-chart-debug').textContent = 
                    `Chart Type: ${chartRecommendations[2].chart_type}\n` +
                    `Config: ${JSON.stringify(mixedChartConfig, null, 2)}`;
            } catch (error) {
                console.error('Error rendering charts:', error);
                alert('Error rendering charts: ' + error.message);
            }
        });
    </script>
</body>
</html>
"""

class ChartTestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)
        
        # Serve the test page
        if parsed_url.path == '/' or parsed_url.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Extract chart rendering functions from chart_viewer.html
            chart_functions = ""
            try:
                with open(os.path.join(os.path.dirname(__file__), 'static', 'chart_viewer.html'), 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract the chart rendering functions
                    start_marker = "function renderChart("
                    end_marker = "function initializeChartViewer()"
                    start_idx = content.find(start_marker)
                    end_idx = content.find(end_marker)
                    
                    if start_idx != -1 and end_idx != -1:
                        chart_functions = content[start_idx:end_idx].strip()
                    else:
                        chart_functions = "// Could not extract chart functions from chart_viewer.html"
                        print("Warning: Could not extract chart functions from chart_viewer.html")
            except Exception as e:
                chart_functions = f"// Error extracting chart functions: {str(e)}"
                print(f"Error extracting chart functions: {str(e)}")
            
            # Replace placeholders in the HTML template
            html_content = HTML_TEMPLATE
            html_content = html_content.replace('SAMPLE_DATA_PLACEHOLDER', json.dumps(SAMPLE_DATA, indent=2))
            html_content = html_content.replace('CHART_RECOMMENDATIONS_PLACEHOLDER', json.dumps(CHART_RECOMMENDATIONS, indent=2))
            html_content = html_content.replace('CHART_FUNCTIONS_PLACEHOLDER', chart_functions)
            
            self.wfile.write(html_content.encode('utf-8'))
            return
        
        # Serve static files
        return super().do_GET()

def find_available_port(start_port=8000, max_attempts=10):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            # Try to bind to the port
            with socketserver.TCPServer(("", port), None) as server:
                pass
            return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

def start_server(port):
    """Start the HTTP server on the specified port."""
    handler = ChartTestHandler
    httpd = socketserver.TCPServer(("", port), handler)
    print(f"Server running at http://localhost:{port}/")
    httpd.serve_forever()

def main():
    # Find an available port
    try:
        port = find_available_port()
        
        # Start the server in a separate thread
        server_thread = threading.Thread(target=start_server, args=(port,))
        server_thread.daemon = True
        server_thread.start()
        
        # Give the server a moment to start
        time.sleep(1)
        
        # Open the browser
        url = f"http://localhost:{port}/"
        print(f"Opening {url} in your browser...")
        webbrowser.open(url)
        
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
