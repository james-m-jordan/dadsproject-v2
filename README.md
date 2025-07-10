# Publishing Analytics Dashboard

An interactive dashboard for modeling and projecting publishing data with 3-year and 5-year forecasts.

## Features

- **Interactive Analysis**: Multiple analysis types including overview, time series, category analysis, and projections
- **Data Modeling**: Trend analysis using linear regression, polynomial regression, and exponential smoothing
- **Projections**: 3-year and 5-year forecasts with multiple methodologies
- **GUI Manipulation**: Interactive controls for filtering, projection methods, and visualization options
- **Export Capabilities**: Download analysis results as CSV files

## Data Structure

The dashboard uses your `bookdataonly.csv` file which contains:
- **Index**: Anonymized identifier combining work references
- **Sales Data**: Annual sales figures from 2016-2024
- **Quantity Data**: Annual quantity sold from 2016-2024
- **Metadata**: Publication info, categories, disciplines, book types

## Dashboard Sections

### 1. Overview
- Portfolio summary metrics
- Key performance indicators
- High-level trend visualizations

### 2. Time Series Analysis
- Historical sales and quantity trends
- Growth rate calculations
- Multi-metric visualization

### 3. Category Analysis
- Performance by book type, discipline, and reporting categories
- Sales distribution and title count analysis
- Category-specific insights

### 4. Projections
- **3-Year Projections**: Forecast through 2027
- **5-Year Projections**: Forecast through 2029
- **Multiple Methods**: Linear, polynomial, and exponential smoothing
- **Detailed Metrics**: Sales, quantities, active titles, revenue per title

### 5. Title Performance
- Top-performing titles analysis
- Performance by publication year
- Individual title insights

## Running the Dashboard

### Option 1: Use the Launcher (Recommended)
```bash
python3 launch_dashboard.py
```

### Option 2: Manual Setup
1. Activate the virtual environment:
   ```bash
   source dashboard_env/bin/activate
   ```

2. Run the dashboard:
   ```bash
   streamlit run publishing_dashboard.py
   ```

### Option 3: Direct Streamlit
```bash
streamlit run publishing_dashboard.py
```

## Dashboard Controls

- **Analysis Type**: Choose between different analysis views
- **Projection Method**: Select modeling approach for forecasts
- **Projection Years**: Set forecast horizon (1-5 years)
- **Data Export**: Download current analysis results

## Key Metrics

- **Total Sales**: Annual revenue trends
- **Active Titles**: Number of titles with sales > 0
- **Revenue per Title**: Average revenue per active title
- **Growth Rates**: Year-over-year percentage changes

## Projection Methodologies

1. **Linear Regression**: Simple trend continuation
2. **Polynomial Regression**: Accounts for non-linear patterns
3. **Exponential Smoothing**: Weighted average with trend adjustment

## Data Privacy

The dashboard maintains data privacy by:
- Using anonymized Index identifiers
- Hiding specific ISBN, author, and title information
- Focusing on aggregated metrics and trends

## Technical Details

- **Framework**: Streamlit for interactive web interface
- **Visualization**: Plotly for interactive charts
- **Modeling**: scikit-learn for predictive analytics
- **Data Processing**: pandas for data manipulation

## Files

- `publishing_dashboard.py`: Main dashboard application
- `launch_dashboard.py`: Launcher script with setup
- `requirements.txt`: Python dependencies
- `dashboard_env/`: Virtual environment directory
- `README.md`: This documentation

## Troubleshooting

If you encounter issues:
1. Ensure all dependencies are installed
2. Check that the CSV file path is correct
3. Verify Python 3.7+ is installed
4. Try running in a virtual environment

## Browser Access

The dashboard will automatically open in your default web browser at:
`http://localhost:8501`