// Mixed Chart Implementation (Bar + Line with dual y-axes)
function renderMixedChart(config, data, divId) {
    console.log(`Rendering mixed chart in ${divId} with config:`, config);
    
    // Validate chart data and configuration
    const validatedData = validateChartData(config, data, divId);
    if (!validatedData) return;
    
    // Use validated data and config
    config = validatedData.config;
    data = validatedData.data;
    
    const xValues = data.map(item => item[config.x_axis]);
    
    // Get all numeric columns for potential y-axes (except the x-axis)
    const numericColumns = Object.keys(data[0]).filter(col => {
        // Skip the x-axis column
        if (col === config.x_axis) return false;
        
        // Check if the column contains numeric data
        const value = data[0][col];
        return typeof value === 'number' || !isNaN(parseFloat(value));
    });
    
    console.log('Potential y-axis columns for mixed chart:', numericColumns);
    
    // Need at least one numeric column for a chart
    if (numericColumns.length < 1) {
        console.warn('No numeric columns for chart, falling back to bar chart');
        return renderBarChart(config, data, divId);
    }
    
    // If we only have one numeric column, render a simple bar chart
    if (numericColumns.length === 1) {
        console.log('Only one numeric column, rendering simple bar chart');
        return renderBarChart(config, data, divId);
    }
    
    // Group columns by scale type
    const monetaryColumns = [];
    const countColumns = [];
    const percentageColumns = [];
    const otherColumns = [];
    
    // Simple heuristic to identify column types based on name
    for (const col of numericColumns) {
        const colLower = col.toLowerCase();
        if (colLower.includes('percent') || colLower.includes('rate') || colLower.includes('ratio')) {
            percentageColumns.push(col);
        } else if (colLower.includes('sales') || colLower.includes('revenue') || 
                   colLower.includes('profit') || colLower.includes('price') || 
                   colLower.includes('cost') || colLower.includes('amount')) {
            monetaryColumns.push(col);
        } else if (colLower.includes('count') || colLower.includes('quantity') || 
                   colLower.includes('number') || colLower.includes('total') || 
                   colLower.includes('sum')) {
            countColumns.push(col);
        } else {
            otherColumns.push(col);
        }
    }
    
    console.log('Column grouping:', {
        monetary: monetaryColumns,
        count: countColumns,
        percentage: percentageColumns,
        other: otherColumns
    });
    
    // Check if we have series configuration from the backend
    let seriesConfig = [];
    if (config.series && Array.isArray(config.series)) {
        seriesConfig = config.series;
        console.log('Using series configuration from backend:', seriesConfig);
    }
    
    // Prepare plot data array
    const plotData = [];
    
    // Color palettes for different groups
    const monetaryColors = ['rgba(17, 157, 255, 0.7)', 'rgba(33, 150, 243, 0.7)', 'rgba(66, 165, 245, 0.7)'];
    const countColors = ['rgba(255, 87, 34, 0.8)', 'rgba(244, 67, 54, 0.8)', 'rgba(239, 83, 80, 0.8)'];
    const percentageColors = ['rgba(76, 175, 80, 0.8)', 'rgba(102, 187, 106, 0.8)', 'rgba(129, 199, 132, 0.8)'];
    const otherColors = ['rgba(156, 39, 176, 0.8)', 'rgba(186, 104, 200, 0.8)', 'rgba(171, 71, 188, 0.8)'];
    
    // Function to create a trace for a column
    function createTrace(column, type, axis, colorArray, colorIndex) {
        const values = data.map(item => {
            const val = item[column];
            return typeof val === 'number' ? val : parseFloat(val);
        });
        
        const color = colorArray[colorIndex % colorArray.length];
        
        return {
            x: xValues,
            y: values,
            name: column,
            type: type,
            yaxis: axis === 'secondary' ? 'y2' : 'y',
            marker: { 
                color: color,
                size: type === 'scatter' ? 8 : undefined
            },
            line: type === 'scatter' ? {
                color: color,
                width: 3
            } : undefined,
            mode: type === 'scatter' ? 'lines+markers' : undefined
        };
    }
    
    // If we have series configuration from backend, use it
    if (seriesConfig.length > 0) {
        seriesConfig.forEach((series, index) => {
            const column = series.column;
            const type = series.type || 'bar';
            const axis = series.axis || 'primary';
            
            // Determine which color palette to use based on column type
            let colorArray;
            let colorIndex;
            
            if (monetaryColumns.includes(column)) {
                colorArray = monetaryColors;
                colorIndex = monetaryColumns.indexOf(column);
            } else if (countColumns.includes(column)) {
                colorArray = countColors;
                colorIndex = countColumns.indexOf(column);
            } else if (percentageColumns.includes(column)) {
                colorArray = percentageColors;
                colorIndex = percentageColumns.indexOf(column);
            } else {
                colorArray = otherColors;
                colorIndex = otherColumns.indexOf(column);
            }
            
            plotData.push(createTrace(column, type, axis, colorArray, colorIndex));
        });
    } else {
        // If no series config, create our own based on column grouping
        
        // Determine if we need a secondary axis
        // Only use secondary axis if we have more than 2 columns with different scale types
        // If we only have 2 columns total, use a single axis for simplicity
        const useSecondaryAxis = numericColumns.length > 2 && 
                                ((monetaryColumns.length > 0 && 
                                 (countColumns.length > 0 || percentageColumns.length > 0)) ||
                                (countColumns.length > 0 && percentageColumns.length > 0));
        
        // Add monetary columns as bars on primary axis
        monetaryColumns.forEach((col, index) => {
            plotData.push(createTrace(col, 'bar', 'primary', monetaryColors, index));
        });
        
        // Add count columns
        countColumns.forEach((col, index) => {
            // If we have monetary columns, put count on secondary axis as lines
            // Otherwise put them as bars on primary axis
            const type = monetaryColumns.length > 0 ? 'scatter' : 'bar';
            const axis = useSecondaryAxis && monetaryColumns.length > 0 ? 'secondary' : 'primary';
            plotData.push(createTrace(col, type, axis, countColors, index));
        });
        
        // Add percentage columns as lines
        percentageColumns.forEach((col, index) => {
            // Always use lines for percentages, but axis depends on what else we have
            const axis = useSecondaryAxis ? 'secondary' : 'primary';
            plotData.push(createTrace(col, 'scatter', axis, percentageColors, index));
        });
        
        // Add other columns
        otherColumns.forEach((col, index) => {
            // Default to bars if we don't have any yet, otherwise lines
            const type = monetaryColumns.length === 0 && countColumns.length === 0 ? 'bar' : 'scatter';
            // Put on secondary axis if we're using it and have other columns there
            const axis = useSecondaryAxis && (countColumns.length > 0 || percentageColumns.length > 0) ? 
                        'secondary' : 'primary';
            plotData.push(createTrace(col, type, axis, otherColors, index));
        });
    }
    
    // Limit to 5 series total for readability
    if (plotData.length > 5) {
        console.warn(`Too many series (${plotData.length}), limiting to 5`);
        plotData.splice(5);
    }
    
    // Determine axis titles based on what we're showing
    let primaryAxisTitle = '';
    let secondaryAxisTitle = '';
    
    // Check if any traces use the primary axis
    const primaryTraces = plotData.filter(trace => !trace.yaxis || trace.yaxis === 'y');
    if (primaryTraces.length > 0) {
        // If all primary traces are the same type, use the first column name
        const primaryTypes = new Set(primaryTraces.map(trace => trace.type));
        if (primaryTypes.size === 1) {
            primaryAxisTitle = primaryTraces[0].name;
        } else {
            // Otherwise use a generic title based on the column types
            if (monetaryColumns.length > 0) {
                primaryAxisTitle = 'Amount ($)';
            } else if (countColumns.length > 0) {
                primaryAxisTitle = 'Count';
            } else {
                primaryAxisTitle = 'Value';
            }
        }
    }
    
    // Check if any traces use the secondary axis
    const secondaryTraces = plotData.filter(trace => trace.yaxis === 'y2');
    if (secondaryTraces.length > 0) {
        // If all secondary traces are the same type, use the first column name
        const secondaryTypes = new Set(secondaryTraces.map(trace => trace.type));
        if (secondaryTypes.size === 1) {
            secondaryAxisTitle = secondaryTraces[0].name;
        } else {
            // Otherwise use a generic title based on the column types
            if (countColumns.length > 0 && secondaryTraces.some(t => countColumns.includes(t.name))) {
                secondaryAxisTitle = 'Count';
            } else if (percentageColumns.length > 0 && secondaryTraces.some(t => percentageColumns.includes(t.name))) {
                secondaryAxisTitle = 'Percentage (%)';
            } else {
                secondaryAxisTitle = 'Value';
            }
        }
    }
    
    // Create layout with appropriate axis configuration
    const layout = {
        title: config.title || 'Multi-metric Analysis',
        xaxis: {
            title: config.x_axis,
            tickangle: -45
        },
        yaxis: {
            title: primaryAxisTitle,
            titlefont: { color: 'rgb(17, 157, 255)' },
            tickfont: { color: 'rgb(17, 157, 255)' }
        },
        margin: {
            l: 60,
            r: 60,
            t: 80,
            b: 120,
            pad: 4
        },
        legend: {
            orientation: 'h',
            y: -0.2
        },
        hovermode: 'closest'
    };
    
    // Add secondary y-axis if needed
    if (secondaryTraces.length > 0) {
        layout.yaxis2 = {
            title: secondaryAxisTitle,
            titlefont: { color: 'rgb(255, 87, 34)' },
            tickfont: { color: 'rgb(255, 87, 34)' },
            overlaying: 'y',
            side: 'right'
        };
    }
    
    // Render the chart
    Plotly.newPlot(divId, plotData, layout);
}
