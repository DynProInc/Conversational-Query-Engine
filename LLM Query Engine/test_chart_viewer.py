#!/usr/bin/env python
"""
Test script to verify chart viewer functionality with sample data
"""
import os
import json
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import time

# Sample data from the user's query
sample_data = {
    "query_output": [
        {
            "QUARTER": 1,
            "TOTAL_SALES": 123047757,
            "TOTAL_QUANTITY_SOLD": 2148484
        },
        {
            "QUARTER": 2,
            "TOTAL_SALES": 150031127,
            "TOTAL_QUANTITY_SOLD": 2404085
        },
        {
            "QUARTER": 3,
            "TOTAL_SALES": 158468348,
            "TOTAL_QUANTITY_SOLD": 2427169
        },
        {
            "QUARTER": 4,
            "TOTAL_SALES": 144055215,
            "TOTAL_QUANTITY_SOLD": 2436600
        }
    ],
    "chart_recommendations": [
        {
            "chart_type": "bar",
            "reasoning": "A bar chart is suitable for comparing total sales and quantity sold across different quarters.",
            "priority": 1,
            "chart_config": {
                "title": "Total Sales and Quantity Sold per Quarter in 2024",
                "x_axis": "QUARTER",
                "y_axis": "TOTAL_SALES",
                "color_by": "QUARTER",
                "chart_library": "plotly"
            }
        },
        {
            "chart_type": "line",
            "reasoning": "A line chart is effective for showing trends over time.",
            "priority": 2,
            "chart_config": {
                "title": "Trend of Total Sales and Quantity Sold per Quarter in 2024",
                "x_axis": "QUARTER",
                "y_axis": "TOTAL_SALES",
                "chart_library": "plotly"
            }
        },
        {
            "chart_type": "pie",
            "reasoning": "A pie chart can be used to show the proportion of total sales contributed by each quarter.",
            "priority": 3,
            "chart_config": {
                "title": "Proportion of Total Sales per Quarter in 2024",
                "x_axis": "QUARTER",
                "y_axis": "TOTAL_SALES",
                "color_by": "QUARTER",
                "chart_library": "plotly"
            }
        },
        {
            "chart_type": "area",
            "reasoning": "An area chart can show the cumulative values over time.",
            "priority": 4,
            "chart_config": {
                "title": "Area Chart of Sales and Quantity Over Quarters",
                "x_axis": "QUARTER",
                "chart_library": "plotly"
            }
        },
        {
            "chart_type": "mixed",
            "reasoning": "A mixed chart can show both sales (bars) and quantity (line) on the same chart with different scales.",
            "priority": 5,
            "chart_config": {
                "title": "Sales and Quantity Comparison with Dual Axis",
                "x_axis": "QUARTER",
                "chart_library": "plotly"
            }
        }
    ]
}

def create_test_html():
    """Create a test HTML file that loads the chart viewer with our sample data"""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chart Viewer Test</title>
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
        h1 {
            color: #333;
        }
        .button-container {
            margin: 20px 0;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
            font-size: 14px;
        }
        button:hover {
            background-color: #45a049;
        }
        iframe {
            width: 100%;
            height: 800px;
            border: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Chart Viewer Test</h1>
        <div class="button-container">
            <button onclick="loadChartViewer()">Load Chart Viewer</button>
            <button onclick="loadChartType('bar')">Test Bar Chart</button>
            <button onclick="loadChartType('line')">Test Line Chart</button>
            <button onclick="loadChartType('pie')">Test Pie Chart</button>
            <button onclick="loadChartType('area')">Test Area Chart</button>
            <button onclick="loadChartType('mixed')">Test Mixed Chart</button>
        </div>
        <iframe id="chartFrame" src="about:blank"></iframe>
    </div>

    <script>
        // Sample data from our test
        const sampleData = SAMPLE_DATA_PLACEHOLDER;
        
        function loadChartViewer() {
            const iframe = document.getElementById('chartFrame');
            iframe.src = 'static/chart_viewer.html';
            
            // Wait for iframe to load
            iframe.onload = function() {
                const iframeWindow = iframe.contentWindow;
                
                // Pass the data to the chart viewer
                if (iframeWindow.executeQuery) {
                    iframeWindow.executeQuery(sampleData);
                } else {
                    console.error('Chart viewer not loaded properly');
                }
            };
        }
        
        function loadChartType(chartType) {
            const iframe = document.getElementById('chartFrame');
            
            // If iframe is not loaded with chart viewer, load it first
            if (iframe.src !== 'static/chart_viewer.html') {
                iframe.src = 'static/chart_viewer.html';
                
                // Wait for iframe to load
                iframe.onload = function() {
                    setTimeout(() => {
                        testChartType(chartType);
                    }, 500);
                };
            } else {
                testChartType(chartType);
            }
        }
        
        function testChartType(chartType) {
            const iframe = document.getElementById('chartFrame');
            const iframeWindow = iframe.contentWindow;
            
            // Find the chart recommendation with the specified type
            const chartRec = sampleData.chart_recommendations.find(rec => rec.chart_type === chartType);
            
            if (!chartRec) {
                console.error(`No chart recommendation found for type: ${chartType}`);
                return;
            }
            
            // Create a test div in the iframe
            const testDiv = iframeWindow.document.createElement('div');
            testDiv.id = 'test-chart';
            testDiv.style.width = '100%';
            testDiv.style.height = '500px';
            
            // Clear any existing content
            iframeWindow.document.body.innerHTML = '';
            iframeWindow.document.body.appendChild(testDiv);
            
            // Render the chart
            try {
                iframeWindow.renderChart(chartRec, sampleData.query_output, 'test-chart');
                console.log(`Successfully rendered ${chartType} chart`);
            } catch (error) {
                console.error(`Error rendering ${chartType} chart:`, error);
                iframeWindow.document.body.innerHTML = `<div style="color: red; padding: 20px;">
                    <h2>Error rendering ${chartType} chart</h2>
                    <pre>${error.toString()}</pre>
                </div>`;
            }
        }
    </script>
</body>
</html>
    """
    
    # Replace the placeholder with actual sample data
    html_content = html_content.replace('SAMPLE_DATA_PLACEHOLDER', json.dumps(sample_data))
    
    # Write to file
    with open('chart_test.html', 'w') as f:
        f.write(html_content)
    
    return os.path.abspath('chart_test.html')

def start_server():
    """Start a simple HTTP server to serve the test files"""
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    print("Starting HTTP server on port 8000...")
    
    # Start the server in a separate thread
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    return httpd

def main():
    """Main function to run the test"""
    # Create the test HTML file
    test_file = create_test_html()
    print(f"Created test file: {test_file}")
    
    # Start the HTTP server
    httpd = start_server()
    
    try:
        # Open the test file in a browser
        test_url = "http://localhost:8000/chart_test.html"
        print(f"Opening {test_url} in browser...")
        webbrowser.open(test_url)
        
        # Keep the server running until user interrupts
        print("Press Ctrl+C to stop the server...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping server...")
        httpd.shutdown()
        print("Server stopped.")

if __name__ == "__main__":
    main()
