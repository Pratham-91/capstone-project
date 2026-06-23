// Configuration for Plotly dark theme
const plotlyLayoutTemplate = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#94a3b8', family: 'Inter, sans-serif' },
    margin: { t: 20, r: 20, l: 50, b: 50 },
    xaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
    yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' }
};

let dashboardData = null;
let currentTicker = null;

// Fetch data and initialize
async function initDashboard() {
    try {
        const response = await fetch('data.json');
        if (!response.ok) throw new Error('Data not ready yet.');
        dashboardData = await response.json();
        
        currentTicker = dashboardData.tickers[0]; // Default to first ticker
        
        // Populate UI
        updateSummaryCards();
        buildStockSelector();
        
        // Render charts
        renderForecastChart();
        renderAllocationChart();
        renderCorrelationChart();
        renderVolatilityChart();
        
    } catch (error) {
        console.error("Failed to load dashboard data:", error);
        setTimeout(initDashboard, 2000); // Retry if file is still being generated
    }
}

function updateSummaryCards() {
    // Find top pick based on highest allocation weight
    let maxWeight = 0;
    let topPick = '';
    
    dashboardData.tickers.forEach(t => {
        if (dashboardData.weights[t] > maxWeight) {
            maxWeight = dashboardData.weights[t];
            topPick = t;
        }
    });
    
    document.getElementById('top-pick').innerText = topPick.split('.')[0];
}

function buildStockSelector() {
    const container = document.getElementById('stock-selector');
    container.innerHTML = '';
    
    dashboardData.tickers.forEach(ticker => {
        const btn = document.createElement('button');
        btn.className = `pill-btn ${ticker === currentTicker ? 'active' : ''}`;
        btn.innerText = ticker.split('.')[0];
        btn.onclick = () => {
            document.querySelectorAll('.pill-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentTicker = ticker;
            renderForecastChart();
        };
        container.appendChild(btn);
    });
}

// 1. Forecast Chart
function renderForecastChart() {
    const testDates = dashboardData.test_dates;
    const actual = dashboardData.forecasts[currentTicker].actual;
    const predicted = dashboardData.forecasts[currentTicker].predicted;
    
    const traceActual = {
        x: testDates,
        y: actual,
        mode: 'lines',
        name: 'Actual Price',
        line: { color: '#94a3b8', width: 2 }
    };
    
    const tracePred = {
        x: testDates,
        y: predicted,
        mode: 'lines',
        name: 'LSTM Predicted',
        line: { color: '#3b82f6', width: 2, dash: 'dot' }
    };
    
    const layout = {
        ...plotlyLayoutTemplate,
        hovermode: 'x unified',
        legend: { orientation: 'h', y: 1.1 }
    };
    
    Plotly.newPlot('forecast-chart', [traceActual, tracePred], layout, {responsive: true, displayModeBar: false});
}

// 2. Portfolio Allocation Chart
function renderAllocationChart() {
    const tickers = dashboardData.tickers.map(t => t.split('.')[0]);
    const amounts = dashboardData.tickers.map(t => dashboardData.allocation[t]);
    
    const trace = {
        labels: tickers,
        values: amounts,
        type: 'pie',
        hole: 0.6,
        marker: {
            colors: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
        },
        textinfo: 'label+percent',
        hoverinfo: 'label+value+percent'
    };
    
    const layout = {
        ...plotlyLayoutTemplate,
        showlegend: false,
        annotations: [{
            font: { size: 20, color: '#f8fafc' },
            showarrow: false,
            text: '₹10L'
        }]
    };
    
    Plotly.newPlot('allocation-chart', [trace], layout, {responsive: true, displayModeBar: false});
}

// 3. Correlation Heatmap
function renderCorrelationChart() {
    const tickers = dashboardData.tickers.map(t => t.split('.')[0]);
    const zData = dashboardData.correlation;
    
    const trace = {
        z: zData,
        x: tickers,
        y: tickers,
        type: 'heatmap',
        colorscale: 'Blues',
        showscale: false,
        textTemplate: "%{z:.2f}",
        textfont: {color: "#ffffff"}
    };
    
    const layout = {
        ...plotlyLayoutTemplate,
        margin: { t: 20, r: 20, l: 80, b: 80 }
    };
    
    Plotly.newPlot('correlation-chart', [trace], layout, {responsive: true, displayModeBar: false});
}

// 4. Volatility Chart
function renderVolatilityChart() {
    const dates = dashboardData.vol_dates;
    const traces = dashboardData.tickers.map((ticker, i) => {
        const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];
        return {
            x: dates,
            y: dashboardData.volatility_history[ticker],
            mode: 'lines',
            name: ticker.split('.')[0],
            line: { width: 1.5, color: colors[i % colors.length] }
        };
    });
    
    const layout = {
        ...plotlyLayoutTemplate,
        hovermode: 'x unified',
        legend: { orientation: 'h', y: 1.1 }
    };
    
    Plotly.newPlot('volatility-chart', traces, layout, {responsive: true, displayModeBar: false});
}

// Sidebar Navigation
document.querySelectorAll('.sidebar nav a').forEach(link => {
    link.addEventListener('click', (e) => {
        document.querySelectorAll('.sidebar nav a').forEach(l => l.classList.remove('active'));
        e.target.classList.add('active');
    });
});

// Refresh Button
document.getElementById('refresh-btn').addEventListener('click', () => {
    initDashboard();
});

// Start
initDashboard();
