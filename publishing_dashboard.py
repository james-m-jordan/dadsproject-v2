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
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .projection-header {
        font-size: 1.5rem;
        color: #2c5282;
        margin-top: 2rem;
        margin-bottom: 1rem;
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
            st.warning("📊 Data file not found. Using demo data for demonstration purposes.")
            return create_demo_data()
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Create Index column combining work reference, title info (anonymized)
        df['Index'] = df['WorkRef'].astype(str) + '_' + df.index.astype(str)
        
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
        st.error(f"Error loading data: {str(e)}")
        st.info("🎭 Loading demo data instead...")
        return create_demo_data()

def validate_uploaded_data(df):
    """Validate that uploaded data has the expected structure"""
    required_cols = ['WorkRef']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check for sales columns
    sales_cols = [col for col in df.columns if 'Net Sales $' in col]
    if not sales_cols:
        return False, "No 'Net Sales $' columns found. Please ensure your CSV has sales data columns."
    
    # Check for quantity columns
    quantity_cols = [col for col in df.columns if 'Net Sales Q' in col]
    if not quantity_cols:
        return False, "No 'Net Sales Q' columns found. Please ensure your CSV has quantity data columns."
    
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
    
    # Aggregate by year
    yearly_data = []
    for year in years:
        sales_col = f'" Net Sales $ \n{year} "'
        qty_col = f'" Net Sales Q\n{year} "'
        
        if sales_col in sales_cols and qty_col in quantity_cols:
            total_sales = df[sales_col].sum()
            total_qty = df[qty_col].sum()
            
            # Count active titles (with sales > 0)
            active_titles = (df[sales_col] > 0).sum()
            
            yearly_data.append({
                'Year': year,
                'Total_Sales': total_sales,
                'Total_Quantity': total_qty,
                'Active_Titles': active_titles,
                'Revenue_Per_Title': total_sales / max(active_titles, 1)
            })
    
    return pd.DataFrame(yearly_data)

def create_projections(yearly_data, method='linear', years_ahead=5):
    """Create projections using different methods"""
    if len(yearly_data) < 3:
        return pd.DataFrame()
    
    X = yearly_data['Year'].values.reshape(-1, 1)
    
    projections = {}
    future_years = [2025, 2026, 2027, 2028, 2029][:years_ahead]
    
    for metric in ['Total_Sales', 'Total_Quantity', 'Active_Titles', 'Revenue_Per_Title']:
        y = yearly_data[metric].values
        
        if method == 'linear':
            model = LinearRegression()
        elif method == 'polynomial':
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ])
        else:  # exponential smoothing
            alpha = 0.3
            smoothed = [y[0]]
            for i in range(1, len(y)):
                smoothed.append(alpha * y[i] + (1 - alpha) * smoothed[-1])
            # Simple trend projection
            trend = (smoothed[-1] - smoothed[-3]) / 2 if len(smoothed) >= 3 else 0
            model = None
        
        if model:
            model.fit(X, y)
            future_X = np.array(future_years).reshape(-1, 1)
            future_y = model.predict(future_X)
            projections[metric] = future_y
        else:
            # Exponential smoothing projection
            last_value = smoothed[-1]
            projections[metric] = [last_value + trend * (i + 1) for i in range(years_ahead)]
    
    proj_df = pd.DataFrame({
        'Year': future_years,
        'Total_Sales': projections['Total_Sales'],
        'Total_Quantity': projections['Total_Quantity'],
        'Active_Titles': projections['Active_Titles'],
        'Revenue_Per_Title': projections['Revenue_Per_Title']
    })
    
    return proj_df

def create_category_analysis(df):
    """Analyze data by publishing categories"""
    category_cols = ['JJ: Reporting Type 2', 'BookType', 'Discipline']
    
    analysis = {}
    for cat_col in category_cols:
        if cat_col in df.columns:
            # Get sales by category
            cat_sales = df.groupby(cat_col)['" Net Sales $ \n2024 "'].sum().sort_values(ascending=False)
            cat_counts = df.groupby(cat_col).size()
            
            analysis[cat_col] = {
                'sales': cat_sales,
                'counts': cat_counts,
                'revenue_per_title': cat_sales / cat_counts
            }
    
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
                df_uploaded['Index'] = df_uploaded['WorkRef'].astype(str) + '_' + df_uploaded.index.astype(str)
                
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
                    st.sidebar.dataframe(df.head(3)[['Index', 'BookType', 'Discipline']].fillna('N/A'))
            
        except Exception as e:
            st.sidebar.error(f"❌ Error processing uploaded file: {str(e)}")
            st.sidebar.error("Please make sure your CSV file has the required columns.")
            df, sales_cols, quantity_cols = load_data()
    else:
        # Use default data loading (demo or local file)
        df, sales_cols, quantity_cols = load_data()
        if df is not None and len(df) == 1000:  # Demo data indicator
            st.sidebar.info("📊 Currently using demo data. Upload your CSV file above to use real data.")
    
    # Analysis options
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Overview", "Time Series Analysis", "Category Analysis", "Projections", "Title Performance"]
    )
    
    # Projection settings
    if analysis_type == "Projections":
        st.sidebar.subheader("Projection Settings")
        projection_method = st.sidebar.selectbox(
            "Projection Method",
            ["linear", "polynomial", "exponential_smoothing"]
        )
        projection_years = st.sidebar.slider("Years to Project", 1, 5, 3)
    
    # Create time series data
    yearly_data = create_time_series_data(df, sales_cols, quantity_cols)
    
    # Data source indicator
    if uploaded_file is not None:
        st.info(f"📊 **Using uploaded data**: {uploaded_file.name} ({len(df):,} titles)")
    elif df is not None and len(df) == 1000:
        st.warning("🎭 **Using demo data** - Upload your CSV file in the sidebar to analyze real data")
    else:
        st.success("📊 **Using local data file**")
    
    # Main dashboard content
    if analysis_type == "Overview":
        st.header("📈 Portfolio Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_titles = len(df)
            st.metric("Total Titles", f"{total_titles:,}")
        
        with col2:
            current_year_sales = yearly_data['Total_Sales'].iloc[-1] if not yearly_data.empty else 0
            st.metric("2024 Sales", f"${current_year_sales:,.0f}")
        
        with col3:
            active_titles = yearly_data['Active_Titles'].iloc[-1] if not yearly_data.empty else 0
            st.metric("Active Titles 2024", f"{active_titles:,}")
        
        with col4:
            revenue_per_title = yearly_data['Revenue_Per_Title'].iloc[-1] if not yearly_data.empty else 0
            st.metric("Revenue per Title", f"${revenue_per_title:,.0f}")
        
        # Overview charts
        if not yearly_data.empty:
            fig1 = px.line(yearly_data, x='Year', y='Total_Sales', 
                          title='Total Sales Trend (2016-2024)',
                          labels={'Total_Sales': 'Sales ($)'})
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
            
            fig2 = px.bar(yearly_data, x='Year', y='Active_Titles',
                         title='Active Titles by Year',
                         labels={'Active_Titles': 'Number of Titles'})
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
    
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
        
        category_analysis = create_category_analysis(df)
        
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
    
    elif analysis_type == "Projections":
        st.header("🔮 Sales Projections")
        
        if not yearly_data.empty:
            # Create projections
            projections = create_projections(yearly_data, projection_method, projection_years)
            
            if not projections.empty:
                # Combine historical and projected data
                combined_data = pd.concat([yearly_data, projections], ignore_index=True)
                combined_data['Data_Type'] = ['Historical'] * len(yearly_data) + ['Projected'] * len(projections)
                
                # Sales projections
                st.subheader("Sales Projections")
                fig = px.line(combined_data, x='Year', y='Total_Sales', color='Data_Type',
                             title=f'Sales Projections ({projection_method.title()} Method)',
                             labels={'Total_Sales': 'Sales ($)'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Projection summary
                st.subheader("Projection Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**3-Year Projection (2027)**")
                    if 2027 in projections['Year'].values:
                        proj_2027 = projections[projections['Year'] == 2027].iloc[0]
                        st.metric("Projected Sales", f"${proj_2027['Total_Sales']:,.0f}")
                        st.metric("Projected Active Titles", f"{proj_2027['Active_Titles']:,.0f}")
                        st.metric("Revenue per Title", f"${proj_2027['Revenue_Per_Title']:,.0f}")
                
                with col2:
                    st.markdown("**5-Year Projection (2029)**")
                    if 2029 in projections['Year'].values:
                        proj_2029 = projections[projections['Year'] == 2029].iloc[0]
                        st.metric("Projected Sales", f"${proj_2029['Total_Sales']:,.0f}")
                        st.metric("Projected Active Titles", f"{proj_2029['Active_Titles']:,.0f}")
                        st.metric("Revenue per Title", f"${proj_2029['Revenue_Per_Title']:,.0f}")
                
                # Detailed projections table
                st.subheader("Detailed Projections")
                proj_display = projections.copy()
                proj_display['Total_Sales'] = proj_display['Total_Sales'].apply(lambda x: f"${x:,.0f}")
                proj_display['Total_Quantity'] = proj_display['Total_Quantity'].apply(lambda x: f"{x:,.0f}")
                proj_display['Active_Titles'] = proj_display['Active_Titles'].apply(lambda x: f"{x:,.0f}")
                proj_display['Revenue_Per_Title'] = proj_display['Revenue_Per_Title'].apply(lambda x: f"${x:,.0f}")
                
                st.dataframe(proj_display, use_container_width=True)
    
    elif analysis_type == "Title Performance":
        st.header("📖 Title Performance Analysis")
        
        # Top performers
        st.subheader("Top Performing Titles (2024)")
        
        top_titles = df.nlargest(20, '" Net Sales $ \n2024 "')[
            ['Index', 'Imprint', 'BookType', 'Discipline', '" Net Sales $ \n2024 "', 'FiscalYearOfPublication']
        ].copy()
        
        top_titles['" Net Sales $ \n2024 "'] = top_titles['" Net Sales $ \n2024 "'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(top_titles, use_container_width=True)
        
        # Performance by publication year
        st.subheader("Performance by Publication Year")
        
        pub_year_analysis = df.groupby('FiscalYearOfPublication').agg({
            '" Net Sales $ \n2024 "': 'sum',
            'Index': 'count'
        }).reset_index()
        
        pub_year_analysis.columns = ['Publication_Year', 'Total_Sales_2024', 'Title_Count']
        pub_year_analysis = pub_year_analysis[pub_year_analysis['Publication_Year'] >= 2010]
        
        fig = px.scatter(pub_year_analysis, x='Publication_Year', y='Total_Sales_2024',
                        size='Title_Count', title='2024 Sales by Publication Year',
                        labels={'Total_Sales_2024': '2024 Sales ($)', 'Publication_Year': 'Publication Year'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Data export option
    st.sidebar.subheader("Data Export")
    if st.sidebar.button("Export Current Analysis"):
        if not yearly_data.empty:
            csv = yearly_data.to_csv(index=False)
            st.sidebar.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"publishing_analysis_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )

if __name__ == "__main__":
    main()