"""
Test script for chart rendering in the Conversational Query Engine.
This script will generate sample data and chart recommendations to test the chart viewer.
"""

import json
import webbrowser
import http.server
import socketserver
import os
from pathlib import Path

# Sample data for testing
sample_data = [
    {"quarter": "Q1", "sales": 120000, "units": 1500, "profit": 35000},
    {"quarter": "Q2", "sales": 150000, "units": 1800, "profit": 42000},
    {"quarter": "Q3", "sales": 180000, "units": 2100, "profit": 51000},
    {"quarter": "Q4", "sales": 210000, "units": 2400, "profit": 63000}
]

# Chart recommendations for testing
chart_recommendations = [
    {
        "chart_type": "line",
        "chart_config": {
            "title": "Quarterly Sales Trend",
            "x_axis": "quarter"
        },
        "reasoning": "Line chart is best for showing trends over time"
    },
    {
        "chart_type": "bar",
        "chart_config": {
            "title": "Quarterly Sales Comparison",
            "x_axis": "quarter"
        },
        "reasoning": "Bar chart is good for comparing values across categories"
    },
    {
        "chart_type": "pie",
        "chart_config": {
            "title": "Profit Distribution by Quarter",
            "values": "profit",
            "labels": "quarter"
        },
        "reasoning": "Pie chart shows the proportion of each category to the whole"
    }
]

# Create a test HTML file that loads chart_viewer.html and tests the charts
def create_test_html():
    test_html_path = Path("test_chart_rendering.html")
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chart Rendering Test</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
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
        }
        pre {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Chart Rendering Test</h1>
        
        <div class="chart-container">
            <h2>Test Data</h2>
            <pre id="test-data"></pre>
        </div>
        
        <div class="chart-container">
            <h2>Chart Recommendations</h2>
            <pre id="chart-recommendations"></pre>
        </div>
        
        <div class="chart-container">
            <h2>Chart Results</h2>
            <div id="charts-container"></div>
        </div>
    </div>

    <script>
        // Load the test data
        const testData = SAMPLE_DATA_PLACEHOLDER;
        document.getElementById('test-data').textContent = JSON.stringify(testData, null, 2);
        
        // Load the chart recommendations
        const chartRecommendations = CHART_RECOMMENDATIONS_PLACEHOLDER;
        document.getElementById('chart-recommendations').textContent = JSON.stringify(chartRecommendations, null, 2);
        
        // Define the chart rendering functions directly
        function renderChart(chartRec, data, divId) {
            console.log(`Attempting to render chart for ${divId}`, chartRec);
            
            // Handle different possible chart recommendation structures
            let chartConfig = {};
            let explicitChartType = null;
            
            // First, try to extract the chart type directly from the recommendation
            if (chartRec.chart_type) {
                explicitChartType = chartRec.chart_type.toLowerCase();
                console.log(`Found explicit chart_type at root level: ${explicitChartType}`);
            }
            
            // Also check if there's a chart_type in the chart_config
            if (chartRec.chart_config && chartRec.chart_config.chart_type) {
                if (!explicitChartType) {
                    explicitChartType = chartRec.chart_config.chart_type.toLowerCase();
                    console.log(`Found chart_type in chart_config: ${explicitChartType}`);
                }
            }
            
            if (chartRec.chart_config) {
                chartConfig = chartRec.chart_config;
                
                // If chart_type is at the root level but not in chart_config, copy it
                if (explicitChartType && !chartConfig.chart_type) {
                    chartConfig.chart_type = explicitChartType;
                    console.log(`Copied chart_type from root level to chart_config: ${explicitChartType}`);
                }
            } else {
                // If no chart_config, create one with the chart_type
                chartConfig = { chart_type: explicitChartType };
                console.log(`Created new chart_config with chart_type: ${explicitChartType}`);
            }
            
            // If we still don't have a chart type, default to bar
            if (!chartConfig.chart_type) {
                console.warn('No chart type specified, defaulting to bar chart');
                chartConfig.chart_type = 'bar';
            }
            
            const chartType = chartConfig.chart_type.toLowerCase();
            console.log(`Rendering chart type: ${chartType}`);
            
            // Normalize chart type to handle various naming conventions
            let normalizedChartType = chartType;
            
            // For exact matches, prioritize them over partial matches
            if (chartType === 'bar' || chartType === 'line' || chartType === 'pie' || 
                chartType === 'scatter' || chartType === 'area' || chartType === 'histogram' || 
                chartType === 'mixed') {
                normalizedChartType = chartType;
                console.log(`Exact match found for chart type: ${normalizedChartType}`);
            }
            // Handle variations in chart type naming with partial matches
            else if (chartType.includes('bar')) normalizedChartType = 'bar';
            else if (chartType.includes('line')) normalizedChartType = 'line';
            else if (chartType.includes('pie') || chartType.includes('donut')) normalizedChartType = 'pie';
            else if (chartType.includes('scatter')) normalizedChartType = 'scatter';
            else if (chartType.includes('area')) normalizedChartType = 'area';
            else if (chartType.includes('histogram')) normalizedChartType = 'histogram';
            else if (chartType.includes('mix') || chartType.includes('combo')) normalizedChartType = 'mixed';
            
            console.log(`Normalized chart type: ${normalizedChartType} (from ${chartType})`);
            
            switch (normalizedChartType) {
                case 'bar':
                    renderBarChart(chartConfig, data, divId);
                    break;
                case 'pie':
                    renderPieChart(chartConfig, data, divId);
                    break;
                case 'line':
                    renderLineChart(chartConfig, data, divId);
                    break;
                case 'scatter':
                    renderScatterChart(chartConfig, data, divId);
                    break;
                case 'area':
                    renderAreaChart(chartConfig, data, divId);
                    break;
                default:
                    console.warn(`Chart type '${normalizedChartType}' not supported, defaulting to bar chart`);
                    renderBarChart(chartConfig, data, divId);
            }
        }
        
        function renderLineChart(config, data, divId) {
            console.log(`Rendering line chart in ${divId} with config:`, config);
            
            const xValues = data.map(item => item[config.x_axis]);
            
            // Get all numeric columns for potential y-axes (except the x-axis)
            const numericColumns = Object.keys(data[0]).filter(col => {
                if (col === config.x_axis) return false;
                const value = data[0][col];
                return typeof value === 'number' || !isNaN(parseFloat(value));
            });
            
            // If y_axis is specified, prioritize it, otherwise use all numeric columns
            const yColumns = config.y_axis ? [config.y_axis] : numericColumns;
            
            // Create a trace for each y column
            const plotData = yColumns.map((col, index) => {
                // Generate a color based on the index
                const colors = ['rgba(17, 157, 255, 0.8)', 'rgba(255, 87, 34, 0.8)', 
                               'rgba(76, 175, 80, 0.8)', 'rgba(156, 39, 176, 0.8)',
                               'rgba(255, 193, 7, 0.8)'];
                const color = colors[index % colors.length];
                
                return {
                    x: xValues,
                    y: data.map(item => {
                        const val = item[col];
                        return typeof val === 'number' ? val : parseFloat(val);
                    }),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: col,
                    marker: { color },
                    line: { color }
                };
            });
            
            const layout = {
                title: config.title || 'Line Chart',
                xaxis: { title: config.x_axis },
                yaxis: { title: yColumns.length === 1 ? yColumns[0] : 'Values' },
                margin: { l: 50, r: 50, b: 100, t: 50, pad: 4 },
                legend: { orientation: 'h', y: -0.2 }
            };
            
            Plotly.newPlot(divId, plotData, layout);
        }
        
        function renderBarChart(config, data, divId) {
            console.log(`Rendering bar chart in ${divId} with config:`, config);
            
            const xValues = data.map(item => item[config.x_axis]);
            
            // Get all numeric columns for potential y-axes (except the x-axis)
            const numericColumns = Object.keys(data[0]).filter(col => {
                if (col === config.x_axis) return false;
                const value = data[0][col];
                return typeof value === 'number' || !isNaN(parseFloat(value));
            });
            
            // If y_axis is specified, prioritize it, otherwise use all numeric columns
            const yColumns = config.y_axis ? [config.y_axis] : numericColumns;
            
            // Create a trace for each y column
            const plotData = yColumns.map((col, index) => {
                // Generate a color based on the index
                const colors = ['rgba(17, 157, 255, 0.8)', 'rgba(255, 87, 34, 0.8)', 
                               'rgba(76, 175, 80, 0.8)', 'rgba(156, 39, 176, 0.8)',
                               'rgba(255, 193, 7, 0.8)'];
                const color = colors[index % colors.length];
                
                return {
                    x: xValues,
                    y: data.map(item => {
                        const val = item[col];
                        return typeof val === 'number' ? val : parseFloat(val);
                    }),
                    type: 'bar',
                    name: col,
                    marker: { color }
                };
            });
            
            const layout = {
                title: config.title || 'Bar Chart',
                xaxis: { title: config.x_axis },
                yaxis: { title: yColumns.length === 1 ? yColumns[0] : 'Values' },
                margin: { l: 50, r: 50, b: 100, t: 50, pad: 4 },
                legend: { orientation: 'h', y: -0.2 }
            };
            
            Plotly.newPlot(divId, plotData, layout);
        }
        
        function renderPieChart(config, data, divId) {
            console.log(`Rendering pie chart in ${divId} with config:`, config);
            
            // For pie charts, we need labels and values
            const labels = config.labels ? data.map(item => item[config.labels]) : data.map(item => item[config.x_axis]);
            
            // Get the values column - could be specified as values, y_axis, or measure
            let valuesColumn = config.values || config.y_axis || config.measure;
            
            // If no values column is specified, try to find a numeric column
            if (!valuesColumn) {
                const numericColumns = Object.keys(data[0]).filter(col => {
                    if (col === config.x_axis || col === config.labels) return false;
                    const value = data[0][col];
                    return typeof value === 'number' || !isNaN(parseFloat(value));
                });
                
                if (numericColumns.length > 0) {
                    valuesColumn = numericColumns[0];
                }
            }
            
            if (!valuesColumn) {
                console.error('No values column found for pie chart');
                return;
            }
            
            const values = data.map(item => {
                const val = item[valuesColumn];
                return typeof val === 'number' ? val : parseFloat(val);
            });
            
            const plotData = [{
                type: 'pie',
                labels: labels,
                values: values,
                textinfo: 'label+percent',
                insidetextorientation: 'radial'
            }];
            
            const layout = {
                title: config.title || 'Pie Chart',
                margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 }
            };
            
            Plotly.newPlot(divId, plotData, layout);
        }
        
        function renderScatterChart(config, data, divId) {
            console.log(`Rendering scatter chart in ${divId} with config:`, config);
            
            const xValues = data.map(item => item[config.x_axis]);
            
            // Get all numeric columns for potential y-axes (except the x-axis)
            const numericColumns = Object.keys(data[0]).filter(col => {
                if (col === config.x_axis) return false;
                const value = data[0][col];
                return typeof value === 'number' || !isNaN(parseFloat(value));
            });
            
            // If y_axis is specified, prioritize it, otherwise use all numeric columns
            const yColumns = config.y_axis ? [config.y_axis] : numericColumns;
            
            // Create a trace for each y column
            const plotData = yColumns.map((col, index) => {
                // Generate a color based on the index
                const colors = ['rgba(17, 157, 255, 0.8)', 'rgba(255, 87, 34, 0.8)', 
                               'rgba(76, 175, 80, 0.8)', 'rgba(156, 39, 176, 0.8)',
                               'rgba(255, 193, 7, 0.8)'];
                const color = colors[index % colors.length];
                
                return {
                    x: xValues,
                    y: data.map(item => {
                        const val = item[col];
                        return typeof val === 'number' ? val : parseFloat(val);
                    }),
                    type: 'scatter',
                    mode: 'markers',
                    name: col,
                    marker: { 
                        color,
                        size: 10
                    }
                };
            });
            
            const layout = {
                title: config.title || 'Scatter Chart',
                xaxis: { title: config.x_axis },
                yaxis: { title: yColumns.length === 1 ? yColumns[0] : 'Values' },
                margin: { l: 50, r: 50, b: 100, t: 50, pad: 4 },
                legend: { orientation: 'h', y: -0.2 }
            };
            
            Plotly.newPlot(divId, plotData, layout);
        }
        
        function renderAreaChart(config, data, divId) {
            console.log(`Rendering area chart in ${divId} with config:`, config);
            
            const xValues = data.map(item => item[config.x_axis]);
            
            // Get all numeric columns for potential y-axes (except the x-axis)
            const numericColumns = Object.keys(data[0]).filter(col => {
                if (col === config.x_axis) return false;
                const value = data[0][col];
                return typeof value === 'number' || !isNaN(parseFloat(value));
            });
            
            // If y_axis is specified, prioritize it, otherwise use all numeric columns
            const yColumns = config.y_axis ? [config.y_axis] : numericColumns;
            
            // Create a trace for each y column
            const plotData = yColumns.map((col, index) => {
                // Generate a color based on the index
                const colors = ['rgba(17, 157, 255, 0.5)', 'rgba(255, 87, 34, 0.5)', 
                               'rgba(76, 175, 80, 0.5)', 'rgba(156, 39, 176, 0.5)',
                               'rgba(255, 193, 7, 0.5)'];
                const color = colors[index % colors.length];
                
                return {
                    x: xValues,
                    y: data.map(item => {
                        const val = item[col];
                        return typeof val === 'number' ? val : parseFloat(val);
                    }),
                    type: 'scatter',
                    mode: 'lines',
                    name: col,
                    fill: 'tozeroy',
                    line: { color },
                    fillcolor: color
                };
            });
            
            const layout = {
                title: config.title || 'Area Chart',
                xaxis: { title: config.x_axis },
                yaxis: { title: yColumns.length === 1 ? yColumns[0] : 'Values' },
                margin: { l: 50, r: 50, b: 100, t: 50, pad: 4 },
                legend: { orientation: 'h', y: -0.2 }
            };
            
            Plotly.newPlot(divId, plotData, layout);
        }
        
        // Wait for the document to be fully loaded
        $(document).ready(function() {
            // Create a mock query response
            const mockResponse = {
                query_output: testData,
                chart_recommendations: chartRecommendations
            };
                
                // Create charts container
                const chartsContainer = document.getElementById('charts-container');
                
                // Process each chart recommendation
                chartRecommendations.forEach((chartRec, index) => {
                    try {
                        // Create a container for this chart
                        const chartDiv = document.createElement('div');
                        chartDiv.className = 'chart-container';
                        
                        // Add a title
                        const titleElem = document.createElement('h3');
                        titleElem.textContent = chartRec.chart_config?.title || `Chart ${index + 1}`;
                        chartDiv.appendChild(titleElem);
                        
                        // Add reasoning if available
                        if (chartRec.reasoning) {
                            const reasoningElem = document.createElement('p');
                            reasoningElem.textContent = `Reasoning: ${chartRec.reasoning}`;
                            chartDiv.appendChild(reasoningElem);
                        }
                        
                        // Create a div for the plot
                        const plotDiv = document.createElement('div');
                        plotDiv.id = `chart-${index}`;
                        plotDiv.className = 'chart';
                        chartDiv.appendChild(plotDiv);
                        
                        // Add debug info
                        const debugInfo = document.createElement('div');
                        debugInfo.className = 'debug-info';
                        debugInfo.textContent = `Chart type: ${chartRec.chart_type || 'unknown'}`;
                        chartDiv.appendChild(debugInfo);
                        
                        chartsContainer.appendChild(chartDiv);
                        
                        // Ensure chart_type is correctly passed to chart_config
                        if (chartRec.chart_type) {
                            if (!chartRec.chart_config) {
                                chartRec.chart_config = { chart_type: chartRec.chart_type };
                            } else if (!chartRec.chart_config.chart_type) {
                                chartRec.chart_config.chart_type = chartRec.chart_type;
                            }
                        }
                        
                        // Render the chart
                        console.log(`Rendering chart ${index + 1}:`, chartRec);
                        renderChart(chartRec, testData, plotDiv.id);
                        
                    } catch (error) {
                        console.error(`Error rendering chart ${index + 1}:`, error);
                        const errorDiv = document.createElement('div');
                        errorDiv.style.color = 'red';
                        errorDiv.textContent = `Error: ${error.message}`;
                        chartsContainer.appendChild(errorDiv);
                    }
                });
                
            }).fail(function(jqxhr, settings, exception) {
                console.error("Failed to load chart viewer script:", exception);
                document.getElementById('charts-container').innerHTML = 
                    "<p style='color:red'>Error loading chart viewer script. Make sure chart_viewer.html is in the static directory.</p>";
            });
        });
    </script>
</body>
</html>
    """
    
    # Replace placeholders with actual data
    html_content = html_content.replace("SAMPLE_DATA_PLACEHOLDER", json.dumps(sample_data))
    html_content = html_content.replace("CHART_RECOMMENDATIONS_PLACEHOLDER", json.dumps(chart_recommendations))
    
    with open(test_html_path, "w") as f:
        f.write(html_content)
    
    return test_html_path

def start_server():
    """Start a simple HTTP server to serve the test HTML file"""
    # Try different ports in case the default is in use
    ports = [8000, 8080, 8888, 9000, 9090]
    
    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create the test HTML file
    test_html_path = create_test_html()
    
    # Try to start the server on different ports
    for PORT in ports:
        try:
            # Start the server
            handler = http.server.SimpleHTTPRequestHandler
            httpd = socketserver.TCPServer(("", PORT), handler)
            
            print(f"Server started at http://localhost:{PORT}")
            print(f"Test page: http://localhost:{PORT}/{test_html_path}")
            
            # Open the test page in the default browser
            webbrowser.open(f"http://localhost:{PORT}/{test_html_path}")
            
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("Server stopped.")
                httpd.server_close()
                return
            
            break  # If we get here, the server started successfully
            
        except OSError as e:
            print(f"Port {PORT} is in use, trying another port...")
            if PORT == ports[-1]:
                print("All ports are in use. Please close some applications and try again.")
                return

if __name__ == "__main__":
    start_server()
