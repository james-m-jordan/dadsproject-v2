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
        # Load the main CSV data
        df = pd.read_csv('/Users/jimjordan/Documents/dadsproject-v2/bookdataonly.csv')
        
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
        return None, None, None

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