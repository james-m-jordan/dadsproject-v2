import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Publishing Analytics Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }
    .projection-header {
        font-size: 1.5rem;
        color: #2c5282;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f4e79;
        color: white;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
    }
    .stButton > button {
        background-color: #1f4e79;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #2c5282;
    }
    .insight-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and process the publishing data"""
    try:
        # Try to load the main CSV data
        csv_paths = [
            '/Users/jimjordan/Documents/dadsproject-v2/bookdataonly.csv',
            'bookdataonly.csv',
            './bookdataonly.csv'
        ]
        
        df = None
        for path in csv_paths:
            try:
                df = pd.read_csv(path)
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            # Create demo data if no file is found
            return create_demo_data()
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Use existing Index column if available, otherwise create from WorkRef
        if 'Index' not in df.columns:
            if 'WorkRef' in df.columns:
                df['Index'] = df['WorkRef'].astype(str) + '_' + df.index.astype(str)
            else:
                df['Index'] = df.index.astype(str)
        
        # Extract year columns for sales data
        sales_cols = [col for col in df.columns if 'Net Sales $' in col and any(str(year) in col for year in range(2016, 2025))]
        quantity_cols = [col for col in df.columns if 'Net Sales Q' in col and any(str(year) in col for year in range(2016, 2025))]
        
        # Clean financial data
        for col in sales_cols:
            df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').str.replace('(', '-').str.replace(')', '').str.replace(' ', '')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        for col in quantity_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Convert dates
        df['PubDate'] = pd.to_datetime(df['PubDate'], errors='coerce')
        df['FiscalYearOfPublication'] = pd.to_numeric(df['FiscalYearOfPublication'], errors='coerce')
        
        return df, sales_cols, quantity_cols
    except Exception as e:
        return create_demo_data()

def validate_uploaded_data(df):
    """Validate that uploaded data has the expected structure"""
    if df.empty:
        return False, "The uploaded file is empty."
    
    required_cols = ['WorkRef']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check for sales columns
    sales_cols = [col for col in df.columns if 'Net Sales $' in col]
    if not sales_cols:
        return False, "No 'Net Sales $' columns found. Please ensure your CSV has sales data columns."
    
    # Check for quantity columns (optional warning)
    quantity_cols = [col for col in df.columns if 'Net Sales Q' in col]
    if not quantity_cols:
        # This is a warning, not an error
        return True, "Data validation passed (Note: No quantity columns found)"
    
    # Check for numeric data in sales columns
    numeric_issues = []
    for col in sales_cols[:3]:  # Check first 3 sales columns
        try:
            pd.to_numeric(df[col].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
        except Exception:
            numeric_issues.append(col)
    
    if numeric_issues:
        return False, f"Sales columns contain non-numeric data: {numeric_issues}"
    
    return True, "Data validation passed"

def create_demo_data():
    """Create demo data for testing purposes"""
    np.random.seed(42)
    
    # Create sample data
    n_titles = 1000
    years = list(range(2016, 2025))
    
    # Generate base data
    titles_data = []
    for i in range(n_titles):
        title_id = f"DEMO_{i+1:04d}"
        pub_year = np.random.choice(range(2010, 2024))
        book_type = np.random.choice(['Monograph', 'Textbook', 'Reference', 'Trade'])
        discipline = np.random.choice(['Economics', 'History', 'Literature', 'Mathematics', 'Political Science', 'Sociology'])
        
        # Generate sales data with some growth trend
        base_sales = np.random.lognormal(6, 1.5)  # Log-normal distribution for realistic sales
        sales_data = {}
        qty_data = {}
        
        for year in years:
            # Add some year-over-year variation and trend
            year_factor = 1 + (year - 2016) * 0.02  # Small growth trend
            variance = np.random.normal(1, 0.3)
            sales = max(0, base_sales * year_factor * variance)
            qty = max(0, int(sales / np.random.uniform(20, 80)))  # Price per unit variation
            
            sales_data[f'" Net Sales $ \n{year} "'] = sales
            qty_data[f'" Net Sales Q\n{year} "'] = qty
        
        title_row = {
            'Index': title_id,
            'WorkRef': f"w{i+1}",
            'BookType': book_type,
            'Discipline': discipline,
            'JJ: Reporting Type 2': f"{np.random.choice(['a. Print', 'b. eBook'])}",
            'FiscalYearOfPublication': pub_year,
            'PubDate': pd.to_datetime(f"{pub_year}-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}"),
            'Imprint': 'Demo University Press',
            **sales_data,
            **qty_data
        }
        titles_data.append(title_row)
    
    df = pd.DataFrame(titles_data)
    
    # Create column lists
    sales_cols = [col for col in df.columns if 'Net Sales $' in col]
    quantity_cols = [col for col in df.columns if 'Net Sales Q' in col]
    
    return df, sales_cols, quantity_cols

def create_time_series_data(df, sales_cols, quantity_cols):
    """Create aggregated time series data"""
    years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    
    # Create mappings of years to actual column names
    sales_year_map = {}
    qty_year_map = {}
    
    for year in years:
        # Find sales column for this year
        sales_col = None
        for col in sales_cols:
            if str(year) in col:
                sales_col = col
                break
        if sales_col:
            sales_year_map[year] = sales_col
        
        # Find quantity column for this year
        qty_col = None
        for col in quantity_cols:
            if str(year) in col:
                qty_col = col
                break
        if qty_col:
            qty_year_map[year] = qty_col
    
    # Aggregate by year
    yearly_data = []
    for year in years:
        if year in sales_year_map:
            sales_col = sales_year_map[year]
            total_sales = df[sales_col].sum()
            
            # Count active titles (with sales > 0)
            active_titles = (df[sales_col] > 0).sum()
            
            # Get quantity data if available
            total_qty = 0
            if year in qty_year_map:
                qty_col = qty_year_map[year]
                total_qty = df[qty_col].sum()
            
            yearly_data.append({
                'Year': year,
                'Total_Sales': total_sales,
                'Total_Quantity': total_qty,
                'Active_Titles': active_titles,
                'Revenue_Per_Title': total_sales / max(active_titles, 1)
            })
    
    return pd.DataFrame(yearly_data)

def create_projections(yearly_data, method='linear', years_ahead=5, confidence_level='Moderate'):
    """Create projections using different methods with confidence level adjustments"""
    if len(yearly_data) < 2:
        return pd.DataFrame()
    
    # Handle edge case where we have minimal data
    if len(yearly_data) == 2 and method == 'polynomial':
        method = 'linear'  # Fall back to linear for insufficient data
    
    X = yearly_data['Year'].values.reshape(-1, 1)
    
    # Confidence level multipliers
    confidence_multipliers = {
        'Conservative': 0.8,  # 20% reduction
        'Moderate': 1.0,      # No change
        'Optimistic': 1.2     # 20% increase
    }
    
    multiplier = confidence_multipliers.get(confidence_level, 1.0)
    
    projections = {}
    future_years = [2025, 2026, 2027, 2028, 2029][:years_ahead]
    
    for metric in ['Total_Sales', 'Total_Quantity', 'Active_Titles', 'Revenue_Per_Title']:
        if metric not in yearly_data.columns:
            continue
            
        y = yearly_data[metric].values
        
        # Skip if all values are zero or NaN
        if np.all(y == 0) or np.all(np.isnan(y)):
            projections[metric] = [0] * years_ahead
            continue
        
        # Clean data
        y = np.nan_to_num(y, nan=0.0)
        
        try:
            if method == 'linear':
                model = LinearRegression()
            elif method == 'polynomial':
                model = Pipeline([
                    ('poly', PolynomialFeatures(degree=min(2, len(y)-1))),
                    ('linear', LinearRegression())
                ])
            else:  # exponential smoothing
                alpha = 0.3
                smoothed = [y[0]]
                for i in range(1, len(y)):
                    smoothed.append(alpha * y[i] + (1 - alpha) * smoothed[-1])
                # Calculate trend more robustly
                if len(smoothed) >= 3:
                    trend = (smoothed[-1] - smoothed[-3]) / 2
                elif len(smoothed) >= 2:
                    trend = smoothed[-1] - smoothed[-2]
                else:
                    trend = 0
                model = None
            
            if model:
                model.fit(X, y)
                future_X = np.array(future_years).reshape(-1, 1)
                future_y = model.predict(future_X)
                
                # Apply confidence level adjustment to growth component
                if len(y) >= 2:
                    base_value = y[-1]  # Last known value
                    growth_component = future_y - base_value
                    adjusted_growth = growth_component * multiplier
                    future_y = base_value + adjusted_growth
                
                # Ensure non-negative predictions for certain metrics
                if metric in ['Total_Sales', 'Total_Quantity', 'Active_Titles']:
                    future_y = np.maximum(future_y, 0)
                projections[metric] = future_y
            else:
                # Exponential smoothing projection with confidence adjustment
                last_value = smoothed[-1]
                adjusted_trend = trend * multiplier
                future_values = []
                for i in range(years_ahead):
                    future_values.append(max(0, last_value + adjusted_trend * (i + 1)))
                projections[metric] = future_values
                
        except Exception as e:
            # Fallback to simple linear trend if model fails
            if len(y) >= 2:
                trend = (y[-1] - y[0]) / (len(y) - 1)
                adjusted_trend = trend * multiplier
                projections[metric] = [max(0, y[-1] + adjusted_trend * (i + 1)) for i in range(years_ahead)]
            else:
                projections[metric] = [y[0]] * years_ahead
    
    # Create DataFrame with available metrics
    proj_dict = {'Year': future_years}
    for metric in ['Total_Sales', 'Total_Quantity', 'Active_Titles', 'Revenue_Per_Title']:
        if metric in projections:
            proj_dict[metric] = projections[metric]
    
    proj_df = pd.DataFrame(proj_dict)
    
    return proj_df

def create_category_analysis(df, sales_cols):
    """Analyze data by publishing categories"""
    category_cols = [
        'JJ: Reporting Type 2', 'BookType', 'Discipline', 'AcquisitionEditor',
        'Edition Status', 'c_ReportingProductType', 'CatalogPosition', 
        'PrincetonLegacyLibrary', 'CoPub', 'Finance - Title counts type III',
        'Finance - IP Book Type', 'HighInvestmentGroup'
    ]
    
    # Find the most recent sales column
    latest_sales_col = None
    if sales_cols:
        # Sort by year and get the latest one
        year_cols = []
        for col in sales_cols:
            for year in range(2024, 2015, -1):  # Check from 2024 down to 2016
                if str(year) in col:
                    year_cols.append((year, col))
                    break
        if year_cols:
            year_cols.sort(reverse=True)
            latest_sales_col = year_cols[0][1]
    
    if not latest_sales_col:
        return {}
    
    analysis = {}
    for cat_col in category_cols:
        if cat_col in df.columns:
            try:
                # Get sales by category
                cat_sales = df.groupby(cat_col)[latest_sales_col].sum().sort_values(ascending=False)
                cat_counts = df.groupby(cat_col).size()
                
                analysis[cat_col] = {
                    'sales': cat_sales,
                    'counts': cat_counts,
                    'revenue_per_title': cat_sales / cat_counts.replace(0, 1)  # Avoid division by zero
                }
            except Exception as e:
                continue
    
    return analysis

def main():
    st.markdown('<h1 class="main-header">📚 Publishing Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df, sales_cols, quantity_cols = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check the file path and format.")
        return
    
    # Sidebar controls
    st.sidebar.header("📊 Dashboard Controls")
    
    # Data upload section
    st.sidebar.subheader("📁 Data Upload")
    st.sidebar.markdown("**Upload your publishing data CSV file**")
    
    # Download template button
    template_data = {
        'WorkRef': ['w1', 'w2', 'w3'],
        'BookType': ['Monograph', 'Textbook', 'Reference'],
        'Discipline': ['Economics', 'History', 'Literature'],
        'JJ: Reporting Type 2': ['a. Print', 'b. eBook', 'a. Print'],
        'FiscalYearOfPublication': [2020, 2021, 2019],
        'PubDate': ['2020-01-15', '2021-03-22', '2019-11-08'],
        'Imprint': ['University Press', 'Academic Press', 'Scholarly Press'],
        '" Net Sales $ \n2022 "': [1500, 2500, 800],
        '" Net Sales $ \n2023 "': [1800, 2800, 950],
        '" Net Sales $ \n2024 "': [2000, 3000, 1200],
        '" Net Sales Q\n2022 "': [25, 50, 15],
        '" Net Sales Q\n2023 "': [30, 55, 18],
        '" Net Sales Q\n2024 "': [35, 60, 22]
    }
    template_df = pd.DataFrame(template_data)
    template_csv = template_df.to_csv(index=False)
    
    st.sidebar.download_button(
        label="📥 Download Template CSV",
        data=template_csv,
        file_name="bookdata_template.csv",
        mime="text/csv",
        help="Download a template CSV file showing the expected format"
    )
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Upload your bookdataonly.csv file to use real data instead of demo data"
    )
    
    # Load data based on upload
    if uploaded_file is not None:
        try:
            # Load uploaded data
            df_uploaded = pd.read_csv(uploaded_file)
            
            # Validate the uploaded data
            is_valid, validation_message = validate_uploaded_data(df_uploaded)
            if not is_valid:
                st.sidebar.error(f"❌ {validation_message}")
                st.sidebar.error("Please upload a properly formatted CSV file.")
                df, sales_cols, quantity_cols = load_data()
            else:
                # Process the uploaded data similar to load_data function
                df_uploaded.columns = df_uploaded.columns.str.strip()
                
                # Use existing Index column if available, otherwise create from WorkRef
                if 'Index' not in df_uploaded.columns:
                    if 'WorkRef' in df_uploaded.columns:
                        df_uploaded['Index'] = df_uploaded['WorkRef'].astype(str) + '_' + df_uploaded.index.astype(str)
                    else:
                        df_uploaded['Index'] = df_uploaded.index.astype(str)
                
                # Extract year columns for sales data
                sales_cols = [col for col in df_uploaded.columns if 'Net Sales $' in col and any(str(year) in col for year in range(2016, 2025))]
                quantity_cols = [col for col in df_uploaded.columns if 'Net Sales Q' in col and any(str(year) in col for year in range(2016, 2025))]
                
                # Clean financial data
                for col in sales_cols:
                    df_uploaded[col] = df_uploaded[col].astype(str).str.replace('$', '').str.replace(',', '').str.replace('(', '-').str.replace(')', '').str.replace(' ', '')
                    df_uploaded[col] = pd.to_numeric(df_uploaded[col], errors='coerce').fillna(0)
                
                for col in quantity_cols:
                    df_uploaded[col] = pd.to_numeric(df_uploaded[col], errors='coerce').fillna(0)
                
                # Convert dates
                df_uploaded['PubDate'] = pd.to_datetime(df_uploaded['PubDate'], errors='coerce')
                df_uploaded['FiscalYearOfPublication'] = pd.to_numeric(df_uploaded['FiscalYearOfPublication'], errors='coerce')
                
                # Use uploaded data
                df, sales_cols, quantity_cols = df_uploaded, sales_cols, quantity_cols
                st.sidebar.success(f"✅ Uploaded data loaded successfully! ({len(df):,} titles)")
                
                # Show data preview option
                if st.sidebar.checkbox("Show Data Preview"):
                    st.sidebar.write("**Sample of uploaded data:**")
                    preview_cols = ['Index']
                    for col in ['BookType', 'Discipline', 'Imprint']:
                        if col in df.columns:
                            preview_cols.append(col)
                    st.sidebar.dataframe(df.head(3)[preview_cols].fillna('N/A'))
                    
                    # Show detected columns
                    st.sidebar.write("**Detected columns:**")
                    st.sidebar.write(f"📊 Sales columns: {len(sales_cols)}")
                    st.sidebar.write(f"📈 Quantity columns: {len(quantity_cols)}")
                    st.sidebar.write(f"📚 Total columns: {len(df.columns)}")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error processing uploaded file: {str(e)}")
            st.sidebar.error("Please make sure your CSV file has the required columns.")
            df, sales_cols, quantity_cols = load_data()
    else:
        # Use default data loading (demo or local file)
        df, sales_cols, quantity_cols = load_data()
        if df is not None and len(df) == 1000:  # Demo data indicator
            st.sidebar.info("📊 Currently using demo data. Upload your CSV file above to use real data.")
    
    # Create time series data
    yearly_data = create_time_series_data(df, sales_cols, quantity_cols)
    
    # Enhanced data source indicator
    if uploaded_file is not None:
        st.markdown('<div class="success-box">📊 <strong>Data Source:</strong> Using uploaded file: ' + uploaded_file.name + f' ({len(df):,} titles)</div>', unsafe_allow_html=True)
    elif df is not None and len(df) == 1000 and uploaded_file is None:
        st.markdown('<div class="warning-box">🎭 <strong>Demo Mode:</strong> Currently using demo data. Upload your CSV file in the sidebar to analyze real data.</div>', unsafe_allow_html=True)
    elif df is not None and uploaded_file is None:
        st.markdown('<div class="success-box">📊 <strong>Data Source:</strong> Using local data file</div>', unsafe_allow_html=True)
    
    # Check if we have any usable data
    if df is None or len(df) == 0:
        st.error("❌ No data available for analysis. Please upload a CSV file.")
        return
    
    if not sales_cols:
        st.warning("⚠️ No sales columns detected in the data. Some analysis features may be limited.")
    
    if yearly_data.empty:
        st.warning("⚠️ No time series data available. Some analysis features may be limited.")
    
    # Create tabs for main navigation
    tab1, tab2 = st.tabs(["📊 Analysis", "🔮 Projections"])
    
    with tab1:
        # Analysis options with better organization
        st.sidebar.subheader("📊 Analysis Options")
        analysis_type = st.sidebar.selectbox(
            "Select Analysis Type",
            ["Overview", "Time Series Analysis", "Category Analysis", "Editorial Analysis", 
             "Format & Product Analysis", "Investment Analysis", "Title Performance"],
            help="Choose the type of analysis you want to perform"
        )
        
        # Add analysis description
        analysis_descriptions = {
            "Overview": "High-level portfolio metrics and trends",
            "Time Series Analysis": "Detailed time-based performance analysis",
            "Category Analysis": "Performance breakdown by categories",
            "Editorial Analysis": "Editor and editorial performance insights",
            "Format & Product Analysis": "Analysis by format and product type",
            "Investment Analysis": "Investment group and co-publishing analysis",
            "Title Performance": "Individual title performance analysis"
        }
        
        if analysis_type in analysis_descriptions:
            st.sidebar.info(f"💡 {analysis_descriptions[analysis_type]}")
        
        # Execute analysis based on type
        if analysis_type == "Overview":
            st.header("📈 Portfolio Overview")
            
            # Key metrics with enhanced display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_titles = len(df)
                st.metric("📚 Total Titles", f"{total_titles:,}")
            
            with col2:
                if not yearly_data.empty:
                    current_year_sales = yearly_data['Total_Sales'].iloc[-1]
                    latest_year = yearly_data['Year'].iloc[-1]
                    # Calculate growth if possible
                    if len(yearly_data) > 1:
                        prev_year_sales = yearly_data['Total_Sales'].iloc[-2]
                        growth = ((current_year_sales - prev_year_sales) / prev_year_sales) * 100
                        st.metric(f"💰 {latest_year} Sales", f"${current_year_sales:,.0f}", f"{growth:+.1f}%")
                    else:
                        st.metric(f"💰 {latest_year} Sales", f"${current_year_sales:,.0f}")
                else:
                    st.metric("💰 Sales", "No data")
            
            with col3:
                if not yearly_data.empty:
                    active_titles = yearly_data['Active_Titles'].iloc[-1]
                    latest_year = yearly_data['Year'].iloc[-1]
                    st.metric(f"🎯 Active Titles {latest_year}", f"{active_titles:,}")
                else:
                    st.metric("🎯 Active Titles", "No data")
            
            with col4:
                if not yearly_data.empty:
                    revenue_per_title = yearly_data['Revenue_Per_Title'].iloc[-1]
                    st.metric("📊 Revenue per Title", f"${revenue_per_title:,.0f}")
                else:
                    st.metric("📊 Revenue per Title", "No data")
            
            # Overview charts with enhanced styling
            if not yearly_data.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = px.line(yearly_data, x='Year', y='Total_Sales', 
                                  title='💰 Total Sales Trend (2016-2024)',
                                  labels={'Total_Sales': 'Sales ($)'})
                    fig1.update_layout(
                        height=400,
                        font=dict(size=12),
                        title_font_size=14,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    fig1.update_traces(line_color='#1f4e79', line_width=3)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    fig2 = px.bar(yearly_data, x='Year', y='Active_Titles',
                                 title='🎯 Active Titles by Year',
                                 labels={'Active_Titles': 'Number of Titles'})
                    fig2.update_layout(
                        height=400,
                        font=dict(size=12),
                        title_font_size=14,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    fig2.update_traces(marker_color='#2c5282')
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Add insights box
                if len(yearly_data) > 1:
                    total_growth = ((yearly_data['Total_Sales'].iloc[-1] - yearly_data['Total_Sales'].iloc[0]) / yearly_data['Total_Sales'].iloc[0]) * 100
                    avg_growth = yearly_data['Total_Sales'].pct_change().mean() * 100
                    st.markdown(f'<div class="insight-box">📈 <strong>Key Insights:</strong> Total sales grew {total_growth:.1f}% over the period. Average annual growth: {avg_growth:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">⚠️ No time series data available for overview charts.</div>', unsafe_allow_html=True)
        
        elif analysis_type == "Time Series Analysis":
            st.header("📊 Time Series Analysis")
            
            if not yearly_data.empty:
                # Multi-metric time series
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Sales Trend', 'Quantity Trend', 'Active Titles', 'Revenue per Title'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                fig.add_trace(go.Scatter(x=yearly_data['Year'], y=yearly_data['Total_Sales'],
                                       mode='lines+markers', name='Total Sales'), row=1, col=1)
                fig.add_trace(go.Scatter(x=yearly_data['Year'], y=yearly_data['Total_Quantity'],
                                       mode='lines+markers', name='Total Quantity'), row=1, col=2)
                fig.add_trace(go.Scatter(x=yearly_data['Year'], y=yearly_data['Active_Titles'],
                                       mode='lines+markers', name='Active Titles'), row=2, col=1)
                fig.add_trace(go.Scatter(x=yearly_data['Year'], y=yearly_data['Revenue_Per_Title'],
                                       mode='lines+markers', name='Revenue per Title'), row=2, col=2)
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Growth rates
                st.subheader("Growth Rates")
                growth_data = yearly_data.copy()
                growth_data['Sales_Growth'] = growth_data['Total_Sales'].pct_change() * 100
                growth_data['Titles_Growth'] = growth_data['Active_Titles'].pct_change() * 100
                
                fig_growth = px.bar(growth_data[1:], x='Year', y=['Sales_Growth', 'Titles_Growth'],
                                   title='Year-over-Year Growth Rates (%)',
                                   barmode='group')
                st.plotly_chart(fig_growth, use_container_width=True)
        
        elif analysis_type == "Category Analysis":
            st.header("📚 Category Analysis")
            
            category_analysis = create_category_analysis(df, sales_cols)
            
            for cat_name, cat_data in category_analysis.items():
                st.subheader(f"Analysis by {cat_name}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if len(cat_data['sales']) > 0:
                        fig = px.bar(x=cat_data['sales'].index, y=cat_data['sales'].values,
                                   title=f'Sales by {cat_name}',
                                   labels={'x': cat_name, 'y': 'Sales ($)'})
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if len(cat_data['counts']) > 0:
                        fig = px.pie(values=cat_data['counts'].values, names=cat_data['counts'].index,
                                   title=f'Title Count Distribution by {cat_name}')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Editorial Analysis":
            st.header("👩‍💼 Editorial Analysis")
            
            if 'AcquisitionEditor' in df.columns and sales_cols:
                # Editor performance analysis
                latest_sales_col = None
                if sales_cols:
                    year_cols = []
                    for col in sales_cols:
                        for year in range(2024, 2015, -1):
                            if str(year) in col:
                                year_cols.append((year, col))
                                break
                    if year_cols:
                        year_cols.sort(reverse=True)
                        latest_sales_col = year_cols[0][1]
                
                if latest_sales_col:
                    st.subheader("Editor Performance Analysis")
                    
                    # Calculate editor metrics
                    editor_metrics = df.groupby('AcquisitionEditor').agg({
                        latest_sales_col: ['sum', 'mean', 'count'],
                        'Index': 'count'
                    }).round(2)
                    
                    editor_metrics.columns = ['Total_Sales', 'Avg_Sales_Per_Title', 'Active_Titles', 'Total_Titles']
                    editor_metrics = editor_metrics.sort_values('Total_Sales', ascending=False).head(15)
                    
                    # Editor performance chart
                    fig = px.bar(
                        x=editor_metrics.index,
                        y=editor_metrics['Total_Sales'],
                        title='Total Sales by Acquisition Editor',
                        labels={'x': 'Editor', 'y': 'Total Sales ($)'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Editor efficiency analysis
                    st.subheader("Editor Efficiency Metrics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.scatter(
                            x=editor_metrics['Total_Titles'],
                            y=editor_metrics['Avg_Sales_Per_Title'],
                            hover_name=editor_metrics.index,
                            title='Titles vs. Average Sales per Title',
                            labels={'x': 'Total Titles', 'y': 'Avg Sales per Title ($)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.write("**Top Editors by Total Sales:**")
                        top_editors = editor_metrics.head(10).copy()
                        top_editors['Total_Sales'] = top_editors['Total_Sales'].apply(lambda x: f"${x:,.0f}")
                        top_editors['Avg_Sales_Per_Title'] = top_editors['Avg_Sales_Per_Title'].apply(lambda x: f"${x:,.0f}")
                        st.dataframe(top_editors)
            
            # Edition status analysis
            if 'Edition Status' in df.columns:
                st.subheader("Edition Status Analysis")
                
                status_counts = df['Edition Status'].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(
                        values=status_counts.values,
                        names=status_counts.index,
                        title='Distribution of Edition Status'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'latest_sales_col' in locals():
                        status_sales = df.groupby('Edition Status')[latest_sales_col].sum().sort_values(ascending=False)
                        fig = px.bar(
                            x=status_sales.index,
                            y=status_sales.values,
                            title='Sales by Edition Status',
                            labels={'x': 'Edition Status', 'y': 'Sales ($)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Format & Product Analysis":
            st.header("📚 Format & Product Analysis")
            
            if 'c_ReportingProductType' in df.columns and sales_cols:
                latest_sales_col = None
                if sales_cols:
                    year_cols = []
                    for col in sales_cols:
                        for year in range(2024, 2015, -1):
                            if str(year) in col:
                                year_cols.append((year, col))
                                break
                    if year_cols:
                        year_cols.sort(reverse=True)
                        latest_sales_col = year_cols[0][1]
                
                if latest_sales_col:
                    st.subheader("Product Format Performance")
                    
                    # Format analysis
                    format_analysis = df.groupby('c_ReportingProductType').agg({
                        latest_sales_col: ['sum', 'mean', 'count'],
                        'Index': 'count'
                    }).round(2)
                    
                    format_analysis.columns = ['Total_Sales', 'Avg_Sales', 'Active_Titles', 'Total_Titles']
                    format_analysis = format_analysis.sort_values('Total_Sales', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.bar(
                            x=format_analysis.index,
                            y=format_analysis['Total_Sales'],
                            title='Sales by Product Format',
                            labels={'x': 'Product Format', 'y': 'Total Sales ($)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.pie(
                            values=format_analysis['Total_Titles'],
                            names=format_analysis.index,
                            title='Title Count by Format'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Format performance table
                    st.subheader("Format Performance Summary")
                    format_display = format_analysis.copy()
                    format_display['Total_Sales'] = format_display['Total_Sales'].apply(lambda x: f"${x:,.0f}")
                    format_display['Avg_Sales'] = format_display['Avg_Sales'].apply(lambda x: f"${x:,.0f}")
                    st.dataframe(format_display)
            
            # Catalog position analysis
            if 'CatalogPosition' in df.columns and sales_cols:
                latest_sales_col = None
                if sales_cols:
                    year_cols = []
                    for col in sales_cols:
                        for year in range(2024, 2015, -1):
                            if str(year) in col:
                                year_cols.append((year, col))
                                break
                    if year_cols:
                        year_cols.sort(reverse=True)
                        latest_sales_col = year_cols[0][1]
                
                if latest_sales_col:
                    st.subheader("Catalog Position Impact")
                    
                    catalog_analysis = df.groupby('CatalogPosition')[latest_sales_col].agg(['sum', 'mean', 'count']).round(2)
                    catalog_analysis.columns = ['Total_Sales', 'Avg_Sales', 'Title_Count']
                    catalog_analysis = catalog_analysis.sort_values('Total_Sales', ascending=False)
                    
                    if not catalog_analysis.empty:
                        fig = px.bar(
                            x=catalog_analysis.index,
                            y=catalog_analysis['Avg_Sales'],
                            title='Average Sales by Catalog Position',
                            labels={'x': 'Catalog Position', 'y': 'Average Sales ($)'}
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Investment Analysis":
            st.header("💰 Investment Analysis")
            
            # Find the most recent sales column for investment analysis
            latest_sales_col = None
            if sales_cols:
                year_cols = []
                for col in sales_cols:
                    for year in range(2024, 2015, -1):
                        if str(year) in col:
                            year_cols.append((year, col))
                            break
                if year_cols:
                    year_cols.sort(reverse=True)
                    latest_sales_col = year_cols[0][1]
            
            if latest_sales_col:
                # Legacy vs Contemporary Analysis
                if 'PrincetonLegacyLibrary' in df.columns:
                    st.subheader("Legacy vs Contemporary Performance")
                    
                    legacy_analysis = df.groupby('PrincetonLegacyLibrary')[latest_sales_col].agg(['sum', 'mean', 'count']).round(2)
                    legacy_analysis.columns = ['Total_Sales', 'Avg_Sales', 'Title_Count']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.bar(
                            x=legacy_analysis.index,
                            y=legacy_analysis['Total_Sales'],
                            title='Total Sales: Legacy vs Contemporary',
                            labels={'x': 'Legacy Library', 'y': 'Total Sales ($)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(
                            x=legacy_analysis.index,
                            y=legacy_analysis['Avg_Sales'],
                            title='Average Sales: Legacy vs Contemporary',
                            labels={'x': 'Legacy Library', 'y': 'Average Sales ($)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # High Investment Group Analysis
                if 'HighInvestmentGroup' in df.columns:
                    st.subheader("Investment Group Performance")
                    
                    investment_analysis = df.groupby('HighInvestmentGroup')[latest_sales_col].agg(['sum', 'mean', 'count']).round(2)
                    investment_analysis.columns = ['Total_Sales', 'Avg_Sales', 'Title_Count']
                    
                    fig = px.scatter(
                        x=investment_analysis['Title_Count'],
                        y=investment_analysis['Avg_Sales'],
                        hover_name=investment_analysis.index,
                        title='Investment Group: Title Count vs Average Sales',
                        labels={'x': 'Title Count', 'y': 'Average Sales ($)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Co-publishing Analysis
                if 'CoPub' in df.columns:
                    st.subheader("Co-Publishing Impact")
                    
                    copub_analysis = df.groupby('CoPub')[latest_sales_col].agg(['sum', 'mean', 'count']).round(2)
                    copub_analysis.columns = ['Total_Sales', 'Avg_Sales', 'Title_Count']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Co-Published Titles", copub_analysis.loc['Yes', 'Title_Count'] if 'Yes' in copub_analysis.index else 0)
                        st.metric("Non Co-Published Titles", copub_analysis.loc['No', 'Title_Count'] if 'No' in copub_analysis.index else 0)
                    
                    with col2:
                        if 'Yes' in copub_analysis.index and 'No' in copub_analysis.index:
                            copub_avg = copub_analysis.loc['Yes', 'Avg_Sales']
                            non_copub_avg = copub_analysis.loc['No', 'Avg_Sales']
                            st.metric("Co-Pub Avg Sales", f"${copub_avg:,.0f}")
                            st.metric("Non Co-Pub Avg Sales", f"${non_copub_avg:,.0f}")
        
        elif analysis_type == "Title Performance":
            st.header("📖 Title Performance Analysis")
            
            # Find the most recent sales column
            latest_sales_col = None
            latest_year = None
            if sales_cols:
                year_cols = []
                for col in sales_cols:
                    for year in range(2024, 2015, -1):
                        if str(year) in col:
                            year_cols.append((year, col))
                            break
                if year_cols:
                    year_cols.sort(reverse=True)
                    latest_year, latest_sales_col = year_cols[0]
            
            if latest_sales_col:
                # Top performers
                st.subheader(f"Top Performing Titles ({latest_year})")
                
                # Select available columns for display
                display_cols = ['Index']
                optional_cols = ['Imprint', 'BookType', 'Discipline', 'FiscalYearOfPublication']
                for col in optional_cols:
                    if col in df.columns:
                        display_cols.append(col)
                display_cols.append(latest_sales_col)
                
                try:
                    top_titles = df.nlargest(20, latest_sales_col)[display_cols].copy()
                    top_titles[latest_sales_col] = top_titles[latest_sales_col].apply(lambda x: f"${x:,.0f}")
                    st.dataframe(top_titles, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying top titles: {str(e)}")
                
                # Performance distribution
                st.subheader("Performance Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(
                        df, 
                        x=latest_sales_col,
                        nbins=30,
                        title=f'Distribution of {latest_year} Sales',
                        labels={latest_sales_col: f'{latest_year} Sales ($)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Top vs bottom quartiles
                    q75 = df[latest_sales_col].quantile(0.75)
                    q25 = df[latest_sales_col].quantile(0.25)
                    
                    quartile_data = pd.DataFrame({
                        'Quartile': ['Top 25%', 'Bottom 25%'],
                        'Count': [
                            (df[latest_sales_col] >= q75).sum(),
                            (df[latest_sales_col] <= q25).sum()
                        ],
                        'Avg_Sales': [
                            df[df[latest_sales_col] >= q75][latest_sales_col].mean(),
                            df[df[latest_sales_col] <= q25][latest_sales_col].mean()
                        ]
                    })
                    
                    fig = px.bar(
                        quartile_data,
                        x='Quartile',
                        y='Avg_Sales',
                        title='Performance Quartiles Comparison',
                        labels={'Avg_Sales': 'Average Sales ($)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Performance by publication year
                if 'FiscalYearOfPublication' in df.columns:
                    st.subheader("Performance by Publication Year")
                    
                    try:
                        pub_year_analysis = df.groupby('FiscalYearOfPublication').agg({
                            latest_sales_col: 'sum',
                            'Index': 'count'
                        }).reset_index()
                        
                        pub_year_analysis.columns = ['Publication_Year', f'Total_Sales_{latest_year}', 'Title_Count']
                        pub_year_analysis = pub_year_analysis[pub_year_analysis['Publication_Year'] >= 2010]
                        
                        if not pub_year_analysis.empty:
                            fig = px.scatter(pub_year_analysis, x='Publication_Year', y=f'Total_Sales_{latest_year}',
                                            size='Title_Count', title=f'{latest_year} Sales by Publication Year',
                                            labels={f'Total_Sales_{latest_year}': f'{latest_year} Sales ($)', 'Publication_Year': 'Publication Year'})
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No publication year data available for analysis.")
                    except Exception as e:
                        st.error(f"Error analyzing publication year data: {str(e)}")
            else:
                st.info("Publication year analysis requires FiscalYearOfPublication column and sales data.")
    
    with tab2:
        # Projections functionality
        st.header("🔮 Sales Projections")
        
        # Projection settings in tab with enhanced UI
        st.subheader("⚙️ Projection Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            projection_years = st.slider("Years to Project", 1, 5, 3, help="Select how many years into the future you want to project")
        
        with col2:
            confidence_level = st.select_slider(
                "Confidence Level",
                options=["Conservative", "Moderate", "Optimistic"],
                value="Moderate",
                help="Choose the confidence level for projections"
            )
        
        # Add projection methodology explanation
        with st.expander("📖 Projection Methodology"):
            st.markdown("""
            **Three projection methods are used:**
            
            1. **Linear Regression**: Assumes constant growth rate, good for stable trends
            2. **Polynomial Regression**: Captures curved patterns, shows acceleration/deceleration
            3. **Exponential Smoothing**: Weights recent data more heavily, adapts to changes
            
            **Confidence Levels:**
            - **Conservative**: Reduces growth projections by 20% (multiplier: 0.8)
            - **Moderate**: Uses baseline projections (multiplier: 1.0)
            - **Optimistic**: Increases growth projections by 20% (multiplier: 1.2)
            
            Each method provides different insights based on your data patterns.
            """)
        
        if not yearly_data.empty:
            # Create projections using all three methods
            methods = ['linear', 'polynomial', 'exponential_smoothing']
            all_projections = []
            
            for method in methods:
                projections = create_projections(yearly_data, method, projection_years, confidence_level)
                if not projections.empty:
                    projections['Method'] = method.replace('_', ' ').title()
                    all_projections.append(projections)
            
            if all_projections:
                # Combine all projections
                combined_projections = pd.concat(all_projections, ignore_index=True)
                
                # Create comprehensive projection chart
                st.subheader("Multi-Method Sales Projections")
                
                # Create figure with historical + all projections
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=yearly_data['Year'],
                    y=yearly_data['Total_Sales'],
                    mode='lines+markers',
                    name='Historical Data',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8)
                ))
                
                # Add projection lines
                colors = ['#ff7f0e', '#2ca02c', '#d62728']
                for i, method in enumerate(['Linear', 'Polynomial', 'Exponential Smoothing']):
                    method_data = combined_projections[combined_projections['Method'] == method]
                    if not method_data.empty:
                        fig.add_trace(go.Scatter(
                            x=method_data['Year'],
                            y=method_data['Total_Sales'],
                            mode='lines+markers',
                            name=f'{method} Projection',
                            line=dict(color=colors[i], width=2, dash='dash'),
                            marker=dict(size=6)
                        ))
                
                fig.update_layout(
                    title='Sales Projections: Multi-Method Comparison',
                    xaxis_title='Year',
                    yaxis_title='Sales ($)',
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Projection summary table
                st.subheader("Projection Summary by Method")
                
                # Create summary table
                summary_data = []
                target_years = [2027, 2029]
                
                for year in target_years:
                    if year <= 2024 + projection_years:
                        year_data = {'Year': year}
                        for method in ['Linear', 'Polynomial', 'Exponential Smoothing']:
                            method_data = combined_projections[
                                (combined_projections['Method'] == method) & 
                                (combined_projections['Year'] == year)
                            ]
                            if not method_data.empty:
                                sales = method_data['Total_Sales'].iloc[0]
                                year_data[f'{method} Sales'] = f"${sales:,.0f}"
                            else:
                                year_data[f'{method} Sales'] = "N/A"
                        summary_data.append(year_data)
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                
                # Method comparison insights with better styling
                st.subheader("🔍 Method Comparison")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.markdown("**📈 Linear Regression**")
                    st.markdown("• Assumes constant growth rate")
                    st.markdown("• Good for stable trends")
                    st.markdown("• Conservative estimates")
                    st.markdown("• Best for: Steady businesses")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.markdown("**📊 Polynomial Regression**")
                    st.markdown("• Captures curve patterns")
                    st.markdown("• Flexible trend fitting")
                    st.markdown("• Shows acceleration/deceleration")
                    st.markdown("• Best for: Changing trends")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.markdown("**🎯 Exponential Smoothing**")
                    st.markdown("• Weights recent data more")
                    st.markdown("• Adapts to trend changes")
                    st.markdown("• Good for volatile data")
                    st.markdown("• Best for: Dynamic markets")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Enhanced analysis
                st.subheader("Enhanced Projection Analysis")
                
                # Show all metrics projections
                metrics_to_show = ['Total_Sales', 'Active_Titles', 'Revenue_Per_Title']
                
                for metric in metrics_to_show:
                    if metric in combined_projections.columns:
                        st.subheader(f"{metric.replace('_', ' ')} Projections")
                        
                        fig = go.Figure()
                        
                        # Add historical data
                        fig.add_trace(go.Scatter(
                            x=yearly_data['Year'],
                            y=yearly_data[metric],
                            mode='lines+markers',
                            name='Historical Data',
                            line=dict(width=3)
                        ))
                        
                        # Add projection lines
                        for i, method in enumerate(['Linear', 'Polynomial', 'Exponential Smoothing']):
                            method_data = combined_projections[combined_projections['Method'] == method]
                            if not method_data.empty:
                                fig.add_trace(go.Scatter(
                                    x=method_data['Year'],
                                    y=method_data[metric],
                                    mode='lines+markers',
                                    name=f'{method} Projection',
                                    line=dict(color=colors[i], width=2, dash='dash')
                                ))
                        
                        fig.update_layout(
                            title=f'{metric.replace("_", " ")} Projections',
                            xaxis_title='Year',
                            yaxis_title=metric.replace('_', ' '),
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Export projections with better UX
                st.subheader("📁 Export Projections")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 Export Projection Data", help="Download all projection data as CSV"):
                        csv = combined_projections.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Projections CSV",
                            data=csv,
                            file_name=f"sales_projections_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                            mime='text/csv'
                        )
                
                with col2:
                    # Add projection summary export
                    if st.button("📋 Export Summary Report", help="Download projection summary as CSV"):
                        if summary_data:
                            summary_csv = summary_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Summary CSV",
                                data=summary_csv,
                                file_name=f"projection_summary_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                                mime='text/csv'
                            )
                
            else:
                st.warning("Unable to generate projections with available data.")
        else:
            st.warning("No time series data available for projections. Please ensure your data contains sales information with years.")
    
    # Enhanced data export options
    st.sidebar.subheader("📁 Data Export")
    
    export_option = st.sidebar.selectbox(
        "Select Export Type",
        ["Time Series Data", "Full Dataset", "Analysis Summary"],
        help="Choose what data to export"
    )
    
    if st.sidebar.button("📊 Export Selected Data"):
        if export_option == "Time Series Data" and not yearly_data.empty:
            csv = yearly_data.to_csv(index=False)
            st.sidebar.download_button(
                label="📥 Download Time Series CSV",
                data=csv,
                file_name=f"time_series_data_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
        elif export_option == "Full Dataset" and df is not None:
            csv = df.to_csv(index=False)
            st.sidebar.download_button(
                label="📥 Download Full Dataset CSV",
                data=csv,
                file_name=f"full_dataset_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
        elif export_option == "Analysis Summary":
            # Create analysis summary
            summary_data = {
                'Metric': ['Total Titles', 'Latest Year Sales', 'Active Titles', 'Revenue per Title'],
                'Value': [
                    len(df) if df is not None else 0,
                    yearly_data['Total_Sales'].iloc[-1] if not yearly_data.empty else 0,
                    yearly_data['Active_Titles'].iloc[-1] if not yearly_data.empty else 0,
                    yearly_data['Revenue_Per_Title'].iloc[-1] if not yearly_data.empty else 0
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            csv = summary_df.to_csv(index=False)
            st.sidebar.download_button(
                label="📥 Download Analysis Summary CSV",
                data=csv,
                file_name=f"analysis_summary_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )

if __name__ == "__main__":
    main()