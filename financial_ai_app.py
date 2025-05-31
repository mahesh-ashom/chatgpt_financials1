import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

# Set page configuration
st.set_page_config(
    page_title="Financial Analysis System",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #4B5563;
    }
    .insight-card {
        background-color: #E8F4FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
    .warning-card {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
    .success-card {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Data loading and preprocessing
@st.cache_data
def load_data(file_path=None, uploaded_file=None):
    """
    Load and preprocess financial data from CSV file
    """
    try:
        # Load data from uploaded file or default path
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        elif file_path is not None:
            df = pd.read_csv(file_path)
        else:
            # Return empty dataframe with expected columns if no file provided
            return pd.DataFrame(columns=['Company', 'Year', 'Quarter', 'Gross Margin', 'Net Profit Margin', 
                                        'ROA', 'ROE', 'Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets'])
        
        # Display success message
        st.success("Data loaded successfully!")
        
        # Clean column names - strip whitespace
        df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
        
        # Drop unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            
        # Basic data cleaning
        # Convert percentage strings to float if needed
        for col in ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE']:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.replace('%', '').astype(float) / 100
        
        # Create Period column for easier filtering
        if 'Year' in df.columns:
            df['Period'] = df['Year'].astype(str)
            if 'Quarter' in df.columns:
                # Handle annual data (Quarter = 0) and quarterly data
                if 'Period_Type' in df.columns:
                    df['Period Type'] = df['Period_Type']
                else:
                    df['Period Type'] = df['Quarter'].apply(lambda q: 'Annual' if q == 0 else 'Quarterly')
                # For quarterly data, add quarter to period
                quarterly_mask = df['Quarter'] > 0
                df.loc[quarterly_mask, 'Period'] = df.loc[quarterly_mask, 'Period'] + '-Q' + df.loc[quarterly_mask, 'Quarter'].astype(str)
            else:
                df['Period Type'] = 'Annual'
        else:
            # If Year column is missing, use Period column if available
            if 'Period' not in df.columns:
                df['Period'] = 'Unknown'
            if 'Period Type' not in df.columns:
                df['Period Type'] = 'Annual'
        
        # Create Date column for time series analysis
        try:
            df['Date'] = df.apply(
                lambda row: datetime(
                    int(row['Year']) if 'Year' in df.columns else 2020, 
                    1 if row.get('Quarter', 0) == 0 else min(((row.get('Quarter', 1) - 1) * 3) + 1, 12), 
                    1
                ), 
                axis=1
            )
        except Exception as e:
            # If date creation fails, create a dummy date column
            st.warning(f"Could not create proper date column: {str(e)}. Using placeholder dates.")
            df['Date'] = pd.date_range(start='1/1/2020', periods=len(df), freq='Q')
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return empty dataframe with expected columns
        return pd.DataFrame(columns=['Company', 'Year', 'Quarter', 'Gross Margin', 'Net Profit Margin', 
                                    'ROA', 'ROE', 'Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets'])

# Portfolio building functions
def build_portfolio(df, criteria):
    """
    Build investment portfolio based on criteria
    """
    try:
        # Ensure required columns exist
        required_cols = ['Company', 'ROE', 'Debt-to-Equity', 'Net Profit Margin']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return pd.DataFrame()
        
        # Get the most recent data for each company
        if 'Date' in df.columns:
            latest_data = df.sort_values('Date', ascending=False).groupby('Company').first().reset_index()
        else:
            # If no Date column, use the data as is
            latest_data = df.groupby('Company').first().reset_index()
        
        # Start with all companies
        portfolio = latest_data.copy()
        
        # Debug information
        st.write(f"Initial portfolio size before filtering: {len(portfolio)}")
        
        # Apply filters with relaxation if needed
        min_roi = criteria.get('min_roi', 0) / 100
        max_risk = criteria.get('max_risk', 3)
        min_growth = criteria.get('min_growth', 0) / 100
        
        # First attempt with strict criteria
        # Use safe filtering with error handling
        try:
            filtered_portfolio = portfolio[
                (portfolio['ROE'] >= min_roi) &
                (portfolio['Debt-to-Equity'] <= max_risk) &
                (portfolio['Net Profit Margin'] >= min_growth)
            ]
        except Exception as e:
            st.error(f"Error filtering portfolio: {str(e)}")
            # Return all companies if filtering fails
            return portfolio
        
        # If no companies match, try relaxed criteria
        if len(filtered_portfolio) == 0:
            st.warning("No companies match the strict criteria. Trying relaxed criteria...")
            
            # Relax criteria by 50%
            relaxed_min_roi = min_roi * 0.5
            relaxed_max_risk = max_risk * 1.5
            relaxed_min_growth = min_growth * 0.5
            
            filtered_portfolio = portfolio[
                (portfolio['ROE'] >= relaxed_min_roi) &
                (portfolio['Debt-to-Equity'] <= relaxed_max_risk) &
                (portfolio['Net Profit Margin'] >= relaxed_min_growth)
            ]
            
            if len(filtered_portfolio) > 0:
                st.success(f"Found {len(filtered_portfolio)} companies with relaxed criteria.")
            else:
                # If still no matches, return all companies with a warning
                st.warning("No companies match even with relaxed criteria. Showing all companies.")
                filtered_portfolio = portfolio
        
        return filtered_portfolio
    
    except Exception as e:
        st.error(f"Error building portfolio: {str(e)}")
        return pd.DataFrame()

# Company analysis functions
def analyze_company(df, company, period_type='Annual', period=None):
    """
    Analyze financial metrics for a specific company
    """
    try:
        # Filter data for the selected company
        company_data = df[df['Company'] == company]
        
        if len(company_data) == 0:
            st.warning(f"No data available for {company}")
            return None
        
        # Filter by period type
        company_data = company_data[company_data['Period Type'] == period_type]
        
        if len(company_data) == 0:
            st.warning(f"No {period_type.lower()} data available for {company}")
            return None
        
        # If period is specified, filter by period
        if period:
            company_data = company_data[company_data['Period'] == period]
            
            if len(company_data) == 0:
                st.warning(f"No data available for {company} in period {period}")
                return None
        
        # Sort by date for time series analysis
        company_data = company_data.sort_values('Date')
        
        return company_data
    
    except Exception as e:
        st.error(f"Error analyzing company: {str(e)}")
        return None

# Company comparison functions
def compare_companies(df, companies, period_type='Annual'):
    """
    Compare financial metrics between companies
    """
    try:
        # Filter data for selected companies and period type
        comparison_data = df[df['Company'].isin(companies) & (df['Period Type'] == period_type)]
        
        if len(comparison_data) == 0:
            st.warning(f"No {period_type.lower()} data available for the selected companies")
            return None
        
        # Sort by date for time series analysis
        comparison_data = comparison_data.sort_values(['Company', 'Date'])
        
        return comparison_data
    
    except Exception as e:
        st.error(f"Error comparing companies: {str(e)}")
        return None

# Industry analysis functions
def analyze_industry(df, period_type='Annual', period=None):
    """
    Analyze industry metrics and benchmarks
    """
    try:
        # Filter data by period type
        industry_data = df[df['Period Type'] == period_type]
        
        if len(industry_data) == 0:
            st.warning(f"No {period_type.lower()} data available")
            return None
        
        # If period is specified, filter by period
        if period:
            industry_data = industry_data[industry_data['Period'] == period]
            
            if len(industry_data) == 0:
                st.warning(f"No data available for period {period}")
                return None
        
        # Calculate industry averages
        industry_avg = industry_data.groupby('Period').mean().reset_index()
        
        return {
            'data': industry_data,
            'averages': industry_avg
        }
    
    except Exception as e:
        st.error(f"Error analyzing industry: {str(e)}")
        return None

# Visualization functions
def plot_time_series(data, metric, title, companies=None):
    """
    Create time series plot for financial metrics
    """
    fig = px.line(
        data, 
        x='Date', 
        y=metric, 
        color='Company' if companies else None,
        title=title,
        labels={metric: metric, 'Date': 'Period'},
        markers=True
    )
    
    fig.update_layout(
        xaxis_title='Period',
        yaxis_title=metric,
        legend_title='Company',
        hovermode='x unified'
    )
    
    # Format y-axis as percentage for certain metrics
    if metric in ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE']:
        fig.update_layout(yaxis_tickformat='.1%')
    
    return fig

def plot_radar_chart(data, metrics, companies):
    """
    Create radar chart for comparing multiple metrics across companies
    """
    fig = go.Figure()
    
    for company in companies:
        company_data = data[data['Company'] == company].iloc[-1]  # Get latest data
        
        fig.add_trace(go.Scatterpolar(
            r=[company_data[m] for m in metrics],
            theta=metrics,
            fill='toself',
            name=company
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # Normalized range
            )
        ),
        title="Financial Metrics Comparison"
    )
    
    return fig

def plot_bar_comparison(data, metric, companies):
    """
    Create bar chart for comparing a single metric across companies
    """
    # Get the latest data for each company
    latest_data = data.sort_values('Date', ascending=False).groupby('Company').first().reset_index()
    
    fig = px.bar(
        latest_data,
        x='Company',
        y=metric,
        color='Company',
        title=f"{metric} Comparison",
        text_auto='.1%' if metric in ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE'] else True
    )
    
    # Format y-axis as percentage for certain metrics
    if metric in ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE']:
        fig.update_layout(yaxis_tickformat='.1%')
    
    return fig

# Main application
def main():
    # Initialize session state for navigation
    if 'page' not in st.session_state:
        st.session_state.page = "Company Analysis"
    
    # Sidebar navigation
    st.sidebar.title("Financial Analysis Options")
    st.sidebar.markdown("---")
    
    st.sidebar.header("Navigation")
    
    # Navigation radio buttons
    page = st.sidebar.radio(
        "Select Analysis Type",
        ["Company Analysis", "Portfolio Builder", "Company Comparison", "Industry Analysis"],
        key="navigation",
        index=["Company Analysis", "Portfolio Builder", "Company Comparison", "Industry Analysis"].index(st.session_state.page)
    )
    
    st.session_state.page = page
    
    # File uploader for CSV data
    uploaded_file = st.sidebar.file_uploader("Upload Financial Data CSV", type=["csv"])
    
    # Load data
    if uploaded_file is not None:
        df = load_data(uploaded_file=uploaded_file)
    else:
        # Try to load from default path
        try:
            df = load_data(file_path="SavolaAlmaraiNADECFinancialRatiosCSV.csv.csv")
        except:
            st.warning("Please upload a CSV file with financial data.")
            df = pd.DataFrame()
    
    # Check if data is loaded
    if df.empty:
        st.error("Failed to load financial data. Please upload a CSV file.")
        return
    
    # Get unique companies, periods, and period types
    companies = sorted(df['Company'].unique())
    period_types = sorted(df['Period Type'].unique())
    periods = sorted(df['Period'].unique())
    
    # Display content based on selected page
    if page == "Company Analysis":
        company_analysis_page(df, companies, period_types, periods)
    
    elif page == "Portfolio Builder":
        portfolio_builder_page(df)
    
    elif page == "Company Comparison":
        company_comparison_page(df, companies, period_types)
    
    elif page == "Industry Analysis":
        industry_analysis_page(df, period_types, periods)

def company_analysis_page(df, companies, period_types, periods):
    """
    Display company analysis page
    """
    st.markdown("<h1 class='main-header'>Company Financial Analysis</h1>", unsafe_allow_html=True)
    
    # Company selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_company = st.selectbox("Select Company", companies)
    
    with col2:
        selected_period_type = st.selectbox("Select Period Type", period_types)
    
    with col3:
        # Filter periods based on selected period type
        filtered_periods = sorted(df[df['Period Type'] == selected_period_type]['Period'].unique())
        selected_period = st.selectbox("Select Period", filtered_periods)
    
    # Analyze selected company
    company_data = analyze_company(df, selected_company, selected_period_type)
    
    if company_data is None:
        return
    
    # Get data for selected period
    period_data = company_data[company_data['Period'] == selected_period]
    
    if period_data.empty:
        st.warning(f"No data available for {selected_company} in {selected_period}")
        return
    
    # Display company overview
    st.markdown(f"<h2 class='sub-header'>{selected_company} Financial Analysis</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3>Period: {selected_period}</h3>", unsafe_allow_html=True)
    
    # Financial metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>ROE</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{period_data['ROE'].values[0]:.1%}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Net Profit Margin</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{period_data['Net Profit Margin'].values[0]:.1%}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Debt-to-Equity</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{period_data['Debt-to-Equity'].values[0]:.2f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # AI-Powered Analysis
    st.markdown("<h2 class='sub-header'>AI-Powered Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Investment recommendation
        roe = period_data['ROE'].values[0]
        npm = period_data['Net Profit Margin'].values[0]
        de = period_data['Debt-to-Equity'].values[0]
        
        # Simple rule-based recommendation
        if roe > 0.1 and npm > 0.05 and de < 2:
            recommendation = "Buy"
            confidence = 75
            st.markdown("<div class='success-card'>", unsafe_allow_html=True)
        elif roe > 0.05 and npm > 0.02 and de < 3:
            recommendation = "Hold"
            confidence = 60
            st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
        else:
            recommendation = "Sell"
            confidence = 65
            st.markdown("<div class='warning-card'>", unsafe_allow_html=True)
        
        st.markdown(f"<h3>Investment Recommendation: {recommendation}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p>Confidence: {confidence}%</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Company status
        if roe > 0.1 and npm > 0.05:
            status = "Good"
            predicted_roe = roe * 1.05  # Simple prediction: 5% growth
        elif roe > 0.05 and npm > 0.02:
            status = "Stable"
            predicted_roe = roe * 1.02  # Simple prediction: 2% growth
        else:
            status = "Concerning"
            predicted_roe = roe * 0.95  # Simple prediction: 5% decline
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<h3>Company Status: {status}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p>Predicted ROE: {predicted_roe:.1%}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Historical Performance
    st.markdown("<h2 class='sub-header'>Historical Performance</h2>", unsafe_allow_html=True)
    
    # Create tabs for different metrics
    tab1, tab2, tab3, tab4 = st.tabs(["ROE Trend", "Profitability", "Liquidity & Leverage", "Statistical Analysis"])
    
    with tab1:
        # ROE trend
        fig = plot_time_series(company_data, 'ROE', f"{selected_company} ROE Trend")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Profitability metrics
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plot_time_series(company_data, 'Gross Margin', f"{selected_company} Gross Margin Trend")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = plot_time_series(company_data, 'Net Profit Margin', f"{selected_company} Net Profit Margin Trend")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Liquidity and leverage metrics
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plot_time_series(company_data, 'Current Ratio', f"{selected_company} Current Ratio Trend")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = plot_time_series(company_data, 'Debt-to-Equity', f"{selected_company} Debt-to-Equity Trend")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Statistical analysis
        st.markdown("<h3>Key Financial Relationships</h3>", unsafe_allow_html=True)
        
        # Calculate correlations in the backend
        corr_matrix = company_data[['ROE', 'Net Profit Margin', 'Debt-to-Equity', 'Current Ratio']].corr()
        
        # Display insights based on correlations
        if abs(corr_matrix.loc['ROE', 'Net Profit Margin']) > 0.5:
            st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
            st.markdown("<p>Strong relationship detected between ROE and Net Profit Margin.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        if abs(corr_matrix.loc['ROE', 'Debt-to-Equity']) > 0.5:
            st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
            st.markdown("<p>Significant impact of leverage (Debt-to-Equity) on ROE detected.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display growth rates
        st.markdown("<h3>Growth Analysis</h3>", unsafe_allow_html=True)
        
        if len(company_data) > 1:
            # Calculate year-over-year growth for key metrics
            first_period = company_data.iloc[0]
            last_period = company_data.iloc[-1]
            
            roe_growth = (last_period['ROE'] - first_period['ROE']) / first_period['ROE'] if first_period['ROE'] != 0 else 0
            npm_growth = (last_period['Net Profit Margin'] - first_period['Net Profit Margin']) / first_period['Net Profit Margin'] if first_period['Net Profit Margin'] != 0 else 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-label'>ROE Growth</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value'>{roe_growth:.1%}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-label'>Net Profit Margin Growth</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value'>{npm_growth:.1%}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

def portfolio_builder_page(df):
    """
    Display portfolio builder page
    """
    st.markdown("<h1 class='main-header'>Portfolio Builder</h1>", unsafe_allow_html=True)
    
    # Investment criteria
    st.markdown("<h2 class='sub-header'>Set Investment Criteria</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_roi = st.slider("Minimum ROI (%)", 0.0, 20.0, 8.0, 0.5)
    
    with col2:
        max_risk = st.slider("Maximum Risk (Debt-to-Equity)", 0.0, 3.0, 1.5, 0.1)
    
    with col3:
        min_growth = st.slider("Minimum Growth (Net Profit Margin %)", 0.0, 15.0, 5.0, 0.5)
    
    # Build portfolio
    criteria = {
        'min_roi': min_roi,
        'max_risk': max_risk,
        'min_growth': min_growth
    }
    
    portfolio = build_portfolio(df, criteria)
    
    # Display portfolio
    st.markdown("<h2 class='sub-header'>Recommended Portfolio</h2>", unsafe_allow_html=True)
    
    if portfolio.empty:
        st.warning("No companies match the selected criteria. Try adjusting your parameters.")
    else:
        # Display portfolio companies
        st.markdown(f"<h3>Selected Companies ({len(portfolio)})</h3>", unsafe_allow_html=True)
        
        # Format percentages for display
        display_portfolio = portfolio.copy()
        for col in ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE']:
            display_portfolio[col] = display_portfolio[col].map('{:.1%}'.format)
        
        # Select columns to display
        display_cols = ['Company', 'ROE', 'Net Profit Margin', 'Debt-to-Equity', 'Current Ratio']
        st.dataframe(display_portfolio[display_cols], use_container_width=True)
        
        # Portfolio visualization
        st.markdown("<h3>Portfolio Visualization</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ROE comparison
            fig = px.bar(
                portfolio,
                x='Company',
                y='ROE',
                color='Company',
                title="Return on Equity (ROE)",
                text_auto='.1%'
            )
            fig.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk-return scatter plot
            fig = px.scatter(
                portfolio,
                x='Debt-to-Equity',
                y='ROE',
                color='Company',
                size='Net Profit Margin',
                size_max=20,
                title="Risk vs. Return",
                labels={'Debt-to-Equity': 'Risk (Debt-to-Equity)', 'ROE': 'Return (ROE)'},
                text='Company'
            )
            fig.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
    
    # Add new company section
    st.markdown("<h2 class='sub-header'>Add New Company</h2>", unsafe_allow_html=True)
    
    st.markdown("Upload CSV file with new company data")
    new_company_file = st.file_uploader("Upload CSV file with new company data", type=["csv"], key="new_company")
    
    if new_company_file is not None:
        st.success("New company data uploaded successfully! The system will include this company in future analyses.")

def company_comparison_page(df, companies, period_types):
    """
    Display company comparison page
    """
    st.markdown("<h1 class='main-header'>Company Comparison</h1>", unsafe_allow_html=True)
    
    # Company selection
    st.markdown("<h2 class='sub-header'>Select Companies to Compare</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        company1 = st.selectbox("Select First Company", companies, key="company1")
    
    with col2:
        # Filter out first company from options
        company2_options = [c for c in companies if c != company1]
        company2 = st.selectbox("Select Second Company", company2_options, key="company2")
    
    with col3:
        selected_period_type = st.selectbox("Select Period Type", period_types, key="period_type_comparison")
    
    # Compare companies
    comparison_data = compare_companies(df, [company1, company2], selected_period_type)
    
    if comparison_data is None:
        return
    
    # Side-by-side metrics
    st.markdown("<h2 class='sub-header'>Financial Metrics Comparison</h2>", unsafe_allow_html=True)
    
    # Get latest data for each company
    latest_data = comparison_data.sort_values('Date', ascending=False).groupby('Company').first().reset_index()
    
    # Create columns for each company
    col1, col2 = st.columns(2)
    
    # Company 1 metrics
    company1_data = latest_data[latest_data['Company'] == company1].iloc[0]
    
    with col1:
        st.markdown(f"<h3>{company1}</h3>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>ROE</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{company1_data['ROE']:.1%}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Net Profit Margin</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{company1_data['Net Profit Margin']:.1%}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Debt-to-Equity</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{company1_data['Debt-to-Equity']:.2f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Company 2 metrics
    company2_data = latest_data[latest_data['Company'] == company2].iloc[0]
    
    with col2:
        st.markdown(f"<h3>{company2}</h3>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>ROE</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{company2_data['ROE']:.1%}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Net Profit Margin</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{company2_data['Net Profit Margin']:.1%}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Debt-to-Equity</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{company2_data['Debt-to-Equity']:.2f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Radar chart comparison
    st.markdown("<h2 class='sub-header'>Multi-Dimensional Comparison</h2>", unsafe_allow_html=True)
    
    # Normalize data for radar chart
    radar_data = latest_data.copy()
    metrics = ['ROE', 'Net Profit Margin', 'ROA', 'Gross Margin']
    
    # Create radar chart
    fig = plot_radar_chart(radar_data, metrics, [company1, company2])
    st.plotly_chart(fig, use_container_width=True)
    
    # Historical trend comparison
    st.markdown("<h2 class='sub-header'>Historical Trend Comparison</h2>", unsafe_allow_html=True)
    
    # Create tabs for different metrics
    tab1, tab2, tab3 = st.tabs(["ROE", "Net Profit Margin", "Debt-to-Equity"])
    
    with tab1:
        fig = plot_time_series(comparison_data, 'ROE', "ROE Comparison", [company1, company2])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = plot_time_series(comparison_data, 'Net Profit Margin', "Net Profit Margin Comparison", [company1, company2])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = plot_time_series(comparison_data, 'Debt-to-Equity', "Debt-to-Equity Comparison", [company1, company2])
        st.plotly_chart(fig, use_container_width=True)
    
    # Comparative insights
    st.markdown("<h2 class='sub-header'>Comparative Insights</h2>", unsafe_allow_html=True)
    
    # ROE comparison
    roe_diff = company1_data['ROE'] - company2_data['ROE']
    roe_diff_pct = roe_diff / company2_data['ROE'] if company2_data['ROE'] != 0 else 0
    
    if roe_diff > 0:
        st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
        st.markdown(f"<p>{company1} has a higher ROE than {company2} by {abs(roe_diff_pct):.1%}.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
        st.markdown(f"<p>{company2} has a higher ROE than {company1} by {abs(roe_diff_pct):.1%}.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Profitability comparison
    npm_diff = company1_data['Net Profit Margin'] - company2_data['Net Profit Margin']
    npm_diff_pct = npm_diff / company2_data['Net Profit Margin'] if company2_data['Net Profit Margin'] != 0 else 0
    
    if npm_diff > 0:
        st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
        st.markdown(f"<p>{company1} has a higher Net Profit Margin than {company2} by {abs(npm_diff_pct):.1%}.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
        st.markdown(f"<p>{company2} has a higher Net Profit Margin than {company1} by {abs(npm_diff_pct):.1%}.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Risk comparison
    de_diff = company1_data['Debt-to-Equity'] - company2_data['Debt-to-Equity']
    de_diff_pct = de_diff / company2_data['Debt-to-Equity'] if company2_data['Debt-to-Equity'] != 0 else 0
    
    if de_diff < 0:
        st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
        st.markdown(f"<p>{company1} has lower financial risk (Debt-to-Equity) than {company2} by {abs(de_diff_pct):.1%}.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
        st.markdown(f"<p>{company2} has lower financial risk (Debt-to-Equity) than {company1} by {abs(de_diff_pct):.1%}.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def industry_analysis_page(df, period_types, periods):
    """
    Display industry analysis page
    """
    st.markdown("<h1 class='main-header'>Industry Analysis</h1>", unsafe_allow_html=True)
    
    # Industry benchmarking
    st.markdown("<h2 class='sub-header'>Industry Benchmarking</h2>", unsafe_allow_html=True)
    
    # Period selection
    col1, col2 = st.columns(2)
    
    with col1:
        selected_period_type = st.selectbox("Select Period Type for Industry Analysis", period_types)
    
    with col2:
        # Filter periods based on selected period type
        filtered_periods = sorted(df[df['Period Type'] == selected_period_type]['Period'].unique())
        selected_period = st.selectbox("Select Period for Industry Analysis", filtered_periods)
    
    # Analyze industry
    industry_data = analyze_industry(df, selected_period_type, selected_period)
    
    if industry_data is None:
        return
    
    # Get companies and industry average
    companies_data = industry_data['data']
    industry_avg = industry_data['averages']
    
    # Industry benchmarks
    st.markdown("<h2 class='sub-header'>Industry Benchmarks</h2>", unsafe_allow_html=True)
    
    # Create tabs for different metrics
    tab1, tab2, tab3 = st.tabs(["Profitability", "Efficiency", "Financial Health"])
    
    with tab1:
        # ROE benchmark
        fig = plot_bar_comparison(companies_data, 'ROE', companies_data['Company'].unique())
        
        # Add industry average line
        avg_roe = companies_data['ROE'].mean()
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(companies_data['Company'].unique()) - 0.5,
            y0=avg_roe,
            y1=avg_roe,
            line=dict(color="red", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=len(companies_data['Company'].unique()) - 1,
            y=avg_roe,
            text=f"Industry Avg: {avg_roe:.1%}",
            showarrow=False,
            yshift=10
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Net Profit Margin benchmark
        fig = plot_bar_comparison(companies_data, 'Net Profit Margin', companies_data['Company'].unique())
        
        # Add industry average line
        avg_npm = companies_data['Net Profit Margin'].mean()
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(companies_data['Company'].unique()) - 0.5,
            y0=avg_npm,
            y1=avg_npm,
            line=dict(color="red", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=len(companies_data['Company'].unique()) - 1,
            y=avg_npm,
            text=f"Industry Avg: {avg_npm:.1%}",
            showarrow=False,
            yshift=10
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # ROA benchmark
        fig = plot_bar_comparison(companies_data, 'ROA', companies_data['Company'].unique())
        
        # Add industry average line
        avg_roa = companies_data['ROA'].mean()
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(companies_data['Company'].unique()) - 0.5,
            y0=avg_roa,
            y1=avg_roa,
            line=dict(color="red", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=len(companies_data['Company'].unique()) - 1,
            y=avg_roa,
            text=f"Industry Avg: {avg_roa:.1%}",
            showarrow=False,
            yshift=10
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Debt-to-Equity benchmark
        fig = plot_bar_comparison(companies_data, 'Debt-to-Equity', companies_data['Company'].unique())
        
        # Add industry average line
        avg_de = companies_data['Debt-to-Equity'].mean()
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(companies_data['Company'].unique()) - 0.5,
            y0=avg_de,
            y1=avg_de,
            line=dict(color="red", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=len(companies_data['Company'].unique()) - 1,
            y=avg_de,
            text=f"Industry Avg: {avg_de:.2f}",
            showarrow=False,
            yshift=10
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Current Ratio benchmark
        fig = plot_bar_comparison(companies_data, 'Current Ratio', companies_data['Company'].unique())
        
        # Add industry average line
        avg_cr = companies_data['Current Ratio'].mean()
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(companies_data['Company'].unique()) - 0.5,
            y0=avg_cr,
            y1=avg_cr,
            line=dict(color="red", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=len(companies_data['Company'].unique()) - 1,
            y=avg_cr,
            text=f"Industry Avg: {avg_cr:.2f}",
            showarrow=False,
            yshift=10
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Company rankings
    st.markdown("<h2 class='sub-header'>Company Rankings</h2>", unsafe_allow_html=True)
    
    # Create rankings
    rankings = companies_data.groupby('Company').mean().reset_index()
    
    # ROE ranking
    roe_ranking = rankings.sort_values('ROE', ascending=False).reset_index(drop=True)
    roe_ranking.index = roe_ranking.index + 1  # Start from 1
    roe_ranking = roe_ranking[['Company', 'ROE']]
    roe_ranking['ROE'] = roe_ranking['ROE'].map('{:.1%}'.format)
    
    # Net Profit Margin ranking
    npm_ranking = rankings.sort_values('Net Profit Margin', ascending=False).reset_index(drop=True)
    npm_ranking.index = npm_ranking.index + 1  # Start from 1
    npm_ranking = npm_ranking[['Company', 'Net Profit Margin']]
    npm_ranking['Net Profit Margin'] = npm_ranking['Net Profit Margin'].map('{:.1%}'.format)
    
    # Display rankings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>ROE Ranking</h3>", unsafe_allow_html=True)
        st.dataframe(roe_ranking, use_container_width=True)
    
    with col2:
        st.markdown("<h3>Net Profit Margin Ranking</h3>", unsafe_allow_html=True)
        st.dataframe(npm_ranking, use_container_width=True)
    
    # Industry insights
    st.markdown("<h2 class='sub-header'>Industry Insights</h2>", unsafe_allow_html=True)
    
    # Calculate industry metrics
    avg_roe = companies_data['ROE'].mean()
    avg_npm = companies_data['Net Profit Margin'].mean()
    avg_de = companies_data['Debt-to-Equity'].mean()
    
    # Identify top performer
    top_performer = roe_ranking.iloc[0]['Company']
    
    # Identify companies above industry average
    above_avg_roe = rankings[rankings['ROE'] > avg_roe]['Company'].tolist()
    above_avg_npm = rankings[rankings['Net Profit Margin'] > avg_npm]['Company'].tolist()
    
    # Display insights
    st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
    st.markdown(f"<p>Top performer in the industry: <strong>{top_performer}</strong></p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
    st.markdown(f"<p>Companies with above-average ROE: <strong>{', '.join(above_avg_roe)}</strong></p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
    st.markdown(f"<p>Companies with above-average Net Profit Margin: <strong>{', '.join(above_avg_npm)}</strong></p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Industry average metrics
    st.markdown("<h3>Industry Average Metrics</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Average ROE</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{avg_roe:.1%}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Average Net Profit Margin</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{avg_npm:.1%}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Average Debt-to-Equity</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{avg_de:.2f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()
