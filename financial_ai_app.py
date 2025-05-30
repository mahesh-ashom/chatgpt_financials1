import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Financial AI Assistant", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-buy {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .recommendation-sell {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .recommendation-hold {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🤖 Financial AI Assistant</h1>', unsafe_allow_html=True)
st.markdown("**Saudi Food Sector Investment Analysis System**")
st.markdown("*Analyze Almarai, Savola, and NADEC with AI-powered insights*")

# Load AI models with comprehensive error handling
@st.cache_resource
def load_ai_models():
    """Load all AI models with fallback options"""
    models = {}
    encoders = {}
    model_status = {}
    
    model_files = {
        'roe_model': 'roe_prediction_model.pkl',
        'investment_model': 'investment_model.pkl', 
        'status_model': 'company_status_model.pkl'
    }
    
    encoder_files = {
        'investment_encoder': 'investment_encoder.pkl',
        'status_encoder': 'status_encoder.pkl',
        'company_encoder': 'company_encoder.pkl'
    }
    
    # Try to load models
    for model_name, filename in model_files.items():
        try:
            models[model_name] = joblib.load(filename)
            model_status[model_name] = "✅ Loaded"
        except Exception as e:
            models[model_name] = None
            model_status[model_name] = f"❌ Failed: {str(e)[:50]}"
    
    # Try to load encoders
    for encoder_name, filename in encoder_files.items():
        try:
            encoders[encoder_name] = joblib.load(filename)
            model_status[encoder_name] = "✅ Loaded"
        except Exception as e:
            encoders[encoder_name] = None
            model_status[encoder_name] = f"❌ Failed: {str(e)[:50]}"
    
    return models, encoders, model_status

# Load financial data with multiple fallback options
@st.cache_data
def load_financial_data():
    """Load and prepare financial data with comprehensive cleaning"""
    try:
        # Try multiple possible filenames
        possible_filenames = [
            'Savola Almarai NADEC Financial Ratios CSV.csv.csv',
            'Savola Almarai NADEC Financial Ratios CSV.csv', 
            'Savola_Almarai_NADEC_Financial_Ratios_CSV.csv',
            'financial_data.csv',
            'data.csv'
        ]
        
        df = None
        loaded_filename = None
        
        for filename in possible_filenames:
            try:
                # Load CSV with minimal processing first
                df = pd.read_csv(filename)
                loaded_filename = filename
                break
            except:
                continue
        
        if df is None:
            # Create sample data if no file found
            st.warning("⚠️ CSV file not found. Using sample data for demonstration.")
            return create_sample_data()
        
        st.success(f"✅ Data loaded from: {loaded_filename}")
        
        # Show RAW data before any cleaning
        st.write("🔍 **RAW CSV DATA (first 3 rows):**")
        st.dataframe(df.head(3))
        
        # Show specific values for debugging
        if 'Gross Margin' in df.columns:
            st.write(f"**Raw Gross Margin values:** {df['Gross Margin'].head(5).tolist()}")
            st.write(f"**Gross Margin data type:** {df['Gross Margin'].dtype}")
        
        # Clean the data
        df = clean_financial_data(df)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_sample_data()

def clean_financial_data(df):
    """FIXED: Just clean the data, keep percentages as percentages"""
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Define columns that should be treated as percentages vs ratios
    percentage_columns = ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE', 'Debt-to-Assets']
    ratio_columns = ['Current Ratio', 'Debt-to-Equity']
    
    # Clean percentage columns - keep them as percentages!
    for col in percentage_columns:
        if col in df.columns:
            # Clean strings (remove % signs, commas, spaces)
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '').str.strip()
            
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # If values are between 0-1, they're decimals - convert to percentages
            if df[col].max() <= 1 and df[col].max() > 0:
                df[col] = df[col] * 100
    
    # Clean ratio columns
    for col in ratio_columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create derived columns if missing
    if 'Period_Type' not in df.columns and 'Period' in df.columns:
        df['Period_Type'] = df['Period'].apply(
            lambda x: 'Annual' if 'Annual' in str(x) or (isinstance(x, (int, float)) and x == int(x)) else 'Quarterly'
        )
    
    if 'Year' not in df.columns and 'Period' in df.columns:
        df['Year'] = df['Period'].str.extract(r'(\d{4})').astype(float).fillna(2023).astype(int)
    
    if 'Quarter' not in df.columns and 'Period' in df.columns:
        df['Quarter'] = df['Period'].str.extract(r'Q(\d)').fillna(0).astype(int)
    
    # Fill missing values with median for each company
    for company in df['Company'].unique():
        company_mask = df['Company'] == company
        for col in percentage_columns + ratio_columns:
            if col in df.columns:
                company_median = df.loc[company_mask, col].median()
                if not pd.isna(company_median):
                    df.loc[company_mask, col] = df.loc[company_mask, col].fillna(company_median)
    
    return df

def create_sample_data():
    """Create sample data for demonstration purposes"""
    companies = ['Almarai', 'Savola', 'NADEC']
    years = [2020, 2021, 2022, 2023]
    quarters = [1, 2, 3, 4]
    
    data = []
    
    # Base financial ratios for each company (realistic Saudi market values in percentages)
    base_ratios = {
        'Almarai': {
            'Gross Margin': 30.9, 'Net Profit Margin': 10.5, 'ROA': 5.7, 'ROE': 11.5,
            'Current Ratio': 1.40, 'Debt-to-Equity': 1.03, 'Debt-to-Assets': 51.0
        },
        'Savola': {
            'Gross Margin': 20.3, 'Net Profit Margin': 4.5, 'ROA': 4.0, 'ROE': 12.6,
            'Current Ratio': 0.84, 'Debt-to-Equity': 2.14, 'Debt-to-Assets': 68.0
        },
        'NADEC': {
            'Gross Margin': 37.0, 'Net Profit Margin': 9.4, 'ROA': 5.9, 'ROE': 8.4,
            'Current Ratio': 1.96, 'Debt-to-Equity': 0.42, 'Debt-to-Assets': 30.0
        }
    }
    
    for company in companies:
        for year in years:
            # Annual data
            annual_ratios = base_ratios[company].copy()
            # Add some year-over-year variation
            for ratio in annual_ratios:
                annual_ratios[ratio] *= (1 + np.random.normal(0, 0.1))
            
            data.append({
                'Company': company,
                'Period': f'{year} Annual',
                'Period_Type': 'Annual',
                'Year': year,
                'Quarter': 0,
                **annual_ratios
            })
            
            # Quarterly data
            for quarter in quarters:
                quarterly_ratios = base_ratios[company].copy()
                # Add quarterly variation
                for ratio in quarterly_ratios:
                    quarterly_ratios[ratio] *= (1 + np.random.normal(0, 0.15))
                
                data.append({
                    'Company': company,
                    'Period': f'{year} Q{quarter}',
                    'Period_Type': 'Quarterly',
                    'Year': year,
                    'Quarter': quarter,
                    **quarterly_ratios
                })
    
    return pd.DataFrame(data)

# Financial AI Class for predictions
class FinancialAI:
    def __init__(self, models, encoders):
        self.models = models
        self.encoders = encoders
        
        # Feature lists (from your original implementation)
        self.roe_features = ['Gross Margin', 'Net Profit Margin', 'ROA', 'Current Ratio',
                            'Debt-to-Equity', 'Debt-to-Assets', 'Year', 'Quarter', 'Company_Encoded']
        self.invest_features = ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE',
                               'Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets',
                               'Year', 'Quarter', 'Company_Encoded']
        self.status_features = ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE',
                               'Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets',
                               'Year', 'Quarter', 'Company_Encoded']
    
    def encode_company(self, company_name):
        """Encode company name"""
        company_mapping = {'Almarai': 0, 'NADEC': 1, 'Savola': 2}
        return company_mapping.get(company_name, 0)
    
    def predict_roe(self, company_data):
        """Predict ROE using the trained model"""
        if self.models['roe_model'] is None:
            return self._fallback_roe_prediction(company_data)
        
        try:
            # Prepare features
            features = []
            for feature in self.roe_features:
                if feature == 'Company_Encoded':
                    features.append(self.encode_company(company_data.get('Company', 'Almarai')))
                else:
                    features.append(company_data.get(feature, 0))
            
            prediction = self.models['roe_model'].predict([features])[0]
            return max(0, min(prediction, 100))  # Bound between 0 and 100%
        except:
            return self._fallback_roe_prediction(company_data)
    
    def predict_investment(self, company_data_with_roe):
        """Predict investment recommendation"""
        if self.models['investment_model'] is None or self.encoders['investment_encoder'] is None:
            return self._fallback_investment_prediction(company_data_with_roe)
        
        try:
            features = []
            for feature in self.invest_features:
                if feature == 'Company_Encoded':
                    features.append(self.encode_company(company_data_with_roe.get('Company', 'Almarai')))
                else:
                    features.append(company_data_with_roe.get(feature, 0))
            
            prediction = self.models['investment_model'].predict([features])[0]
            recommendation = self.encoders['investment_encoder'].inverse_transform([prediction])[0]
            confidence = max(self.models['investment_model'].predict_proba([features])[0])
            
            return recommendation, confidence
        except:
            return self._fallback_investment_prediction(company_data_with_roe)
    
    def predict_status(self, company_data_with_roe):
        """Predict company status"""
        if self.models['status_model'] is None or self.encoders['status_encoder'] is None:
            return self._fallback_status_prediction(company_data_with_roe)
        
        try:
            features = []
            for feature in self.status_features:
                if feature == 'Company_Encoded':
                    features.append(self.encode_company(company_data_with_roe.get('Company', 'Almarai')))
                else:
                    features.append(company_data_with_roe.get(feature, 0))
            
            prediction = self.models['status_model'].predict([features])[0]
            status = self.encoders['status_encoder'].inverse_transform([prediction])[0]
            
            return status
        except:
            return self._fallback_status_prediction(company_data_with_roe)
    
    def _fallback_roe_prediction(self, company_data):
        """Fallback ROE prediction using financial relationships (UPDATED for percentage values)"""
        roa = company_data.get('ROA', 5.0)  # Now in percentage format
        npm = company_data.get('Net Profit Margin', 8.0)  # Now in percentage format
        equity_multiplier = 1 + company_data.get('Debt-to-Equity', 0.8)
        
        # ROE = ROA × Equity Multiplier (simplified DuPont formula)
        predicted_roe = roa * equity_multiplier
        
        # Alternative calculation using profit margin relationship
        if predicted_roe == 0:
            predicted_roe = npm * 1.2  # Rough approximation
        
        return max(0, min(predicted_roe, 100))  # Return as percentage
    
    def _fallback_investment_prediction(self, company_data):
        """Fallback investment prediction using scoring system"""
        score = self._calculate_investment_score(company_data)
        
        if score >= 70:
            return "Strong Buy", 0.85
        elif score >= 60:
            return "Buy", 0.75
        elif score >= 40:
            return "Hold", 0.65
        else:
            return "Sell", 0.55
    
    def _fallback_status_prediction(self, company_data):
        """Fallback status prediction (UPDATED for percentage values)"""
        roe = company_data.get('ROE', 0)
        npm = company_data.get('Net Profit Margin', 0)
        
        if roe > 15 and npm > 15:  # Now using percentage thresholds
            return "Excellent"
        elif roe > 10 and npm > 10:
            return "Good"
        elif roe > 5 and npm > 5:
            return "Average"
        else:
            return "Poor"
    
    def _calculate_investment_score(self, data):
        """Calculate investment score based on financial metrics (UPDATED for percentage values)"""
        score = 0
        
        # ROE scoring (35% weight) - now using percentage values
        roe = data.get('ROE', 0)
        if roe > 20: score += 35
        elif roe > 15: score += 30
        elif roe > 12: score += 25
        elif roe > 8: score += 15
        elif roe > 5: score += 8
        elif roe > 2: score += 3
        
        # ROA scoring (25% weight) - now using percentage values
        roa = data.get('ROA', 0)
        if roa > 12: score += 25
        elif roa > 8: score += 20
        elif roa > 6: score += 15
        elif roa > 4: score += 10
        elif roa > 2: score += 5
        
        # Net Profit Margin scoring (20% weight) - now using percentage values
        npm = data.get('Net Profit Margin', 0)
        if npm > 15: score += 20
        elif npm > 10: score += 15
        elif npm > 5: score += 10
        elif npm > 2: score += 5
        
        # Current Ratio scoring (10% weight)
        cr = data.get('Current Ratio', 0)
        if cr > 2.0: score += 10
        elif cr > 1.5: score += 8
        elif cr > 1.2: score += 6
        elif cr > 1.0: score += 3
        
        # Debt-to-Equity scoring (10% weight)
        de = data.get('Debt-to-Equity', 0)
        if de < 0.3: score += 10
        elif de < 0.5: score += 8
        elif de < 0.8: score += 6
        elif de < 1.2: score += 4
        elif de < 1.5: score += 2
        
        return score
    
    def comprehensive_analysis(self, company_data):
        """Perform comprehensive financial analysis"""
        # Predict ROE
        predicted_roe = self.predict_roe(company_data)
        
        # Update data with predicted ROE
        company_data_with_roe = company_data.copy()
        company_data_with_roe['ROE'] = predicted_roe
        
        # Get predictions
        investment_rec, confidence = self.predict_investment(company_data_with_roe)
        company_status = self.predict_status(company_data_with_roe)
        
        return {
            'predicted_roe': predicted_roe,
            'investment_recommendation': investment_rec,
            'investment_confidence': confidence,
            'company_status': company_status,
            'investment_score': self._calculate_investment_score(company_data_with_roe)
        }

# Load data and models
df = load_financial_data()
models, encoders, model_status = load_ai_models()

# Initialize AI system
financial_ai = FinancialAI(models, encoders)

# Debug section (outside cached function)
if st.sidebar.checkbox("🔍 Show Data Debug Info", value=True):
    st.sidebar.subheader("📊 Data Debug Information")
    
    if not df.empty:
        st.sidebar.write(f"**Total rows loaded:** {len(df)}")
        st.sidebar.write(f"**Columns:** {list(df.columns)}")
        st.sidebar.write(f"**Companies:** {list(df['Company'].unique())}")
        
        # Show first few rows of raw data
        st.sidebar.write("**First 3 rows of data:**")
        st.sidebar.dataframe(df.head(3))
        
        # Show sample data for Almarai 2023 to verify
        almarai_2023 = df[(df['Company'] == 'Almarai') & (df['Year'] == 2023) & (df['Period_Type'] == 'Annual')]
        st.sidebar.write(f"**Almarai 2023 Annual rows found:** {len(almarai_2023)}")
        
        if not almarai_2023.empty:
            sample_row = almarai_2023.iloc[0]
            st.sidebar.write("**Almarai 2023 Annual Values:**")
            st.sidebar.write(f"• Gross Margin: {sample_row.get('Gross Margin', 0):.1f}%")
            st.sidebar.write(f"• Net Profit Margin: {sample_row.get('Net Profit Margin', 0):.1f}%")
            st.sidebar.write(f"• ROE: {sample_row.get('ROE', 0):.1f}%")
            st.sidebar.write(f"• ROA: {sample_row.get('ROA', 0):.1f}%")
            st.sidebar.write(f"• Current Ratio: {sample_row.get('Current Ratio', 0):.2f}")
            st.sidebar.write(f"• Debt-to-Equity: {sample_row.get('Debt-to-Equity', 0):.2f}")
            
            # Show raw values for debugging
            st.sidebar.write("**Raw Values:**")
            raw_row = almarai_2023.iloc[0]
            st.sidebar.write(f"• Gross Margin Raw: {raw_row.get('Gross Margin', 0):.4f}")
            st.sidebar.write(f"• Expected: 30.9 for 30.9%")
        else:
            st.sidebar.error("❌ No Almarai 2023 Annual data found!")
    else:
        st.sidebar.error("❌ No data loaded at all!")

# Sidebar for navigation and model status
st.sidebar.title("🎯 Navigation")

# Model status display
with st.sidebar.expander("🤖 Model Status", expanded=False):
    for model_name, status in model_status.items():
        st.write(f"**{model_name}:** {status}")

# Main navigation
page = st.sidebar.selectbox(
    "Choose Analysis Type:",
    ["🏠 Dashboard", "📊 Company Analysis", "🔮 Ratio Prediction", "🏥 Health Check", "⚖️ Comparison", "🎯 Custom Analysis"]
)

# Dashboard Page
if page == "🏠 Dashboard":
    st.header("📊 Financial AI Dashboard")
    st.markdown("*Overview of Saudi Food Sector Performance*")
    
    if not df.empty:
        # Key metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_companies = df['Company'].nunique()
            st.metric("Companies Analyzed", total_companies)
        
        with col2:
            date_range = f"{df['Year'].min()}-{df['Year'].max()}"
            st.metric("Data Period", date_range)
        
        with col3:
            total_records = len(df)
            st.metric("Financial Records", total_records)
        
        with col4:
            avg_roe = df['ROE'].mean()
            st.metric("Avg Sector ROE", f"{avg_roe:.1f}%")  # FIXED: Show as percentage
        
        # Latest performance summary
        st.subheader("🏆 Latest Company Performance")
        
        # Get latest data for each company
        latest_year = df['Year'].max()
        latest_data = df[(df['Year'] == latest_year) & (df['Period_Type'] == 'Annual')]
        
        if not latest_data.empty:
            # Create performance comparison
            performance_cols = st.columns(len(latest_data))
            
            for i, (_, company_data) in enumerate(latest_data.iterrows()):
                with performance_cols[i]:
                    company = company_data['Company']
                    roe = company_data['ROE']
                    
                    # Generate recommendation
                    recommendation, confidence = financial_ai.predict_investment(company_data)
                    
                    st.markdown(f"### {company}")
                    st.metric("ROE", f"{roe:.1f}%")  # FIXED: Show as percentage
                    
                    # Color-coded recommendation
                    if recommendation in ["Strong Buy", "Buy"]:
                        st.success(f"📈 {recommendation}")
                    elif "Hold" in recommendation:
                        st.warning(f"⚖️ {recommendation}")
                    else:
                        st.error(f"📉 {recommendation}")
        
        # Sector trends chart
        st.subheader("📈 Sector ROE Trends")
        
        # Create trend chart
        annual_data = df[df['Period_Type'] == 'Annual'].groupby(['Year', 'Company'])['ROE'].mean().reset_index()
        
        if not annual_data.empty:
            fig = px.line(
                annual_data, 
                x='Year', 
                y='ROE', 
                color='Company',
                title="ROE Performance Over Time",
                labels={'ROE': 'Return on Equity (%)', 'Year': 'Year'},
                markers=True
            )
            fig.update_layout(yaxis_title="ROE (%)")  # FIXED: Chart axis
            st.plotly_chart(fig, use_container_width=True)

# Company Analysis Page
elif page == "📊 Company Analysis":
    st.header("📊 Individual Company Analysis")
    st.markdown("*Deep dive into specific company performance*")
    
    if not df.empty:
        # Company selection
        available_companies = sorted(df['Company'].unique())
        company = st.selectbox("Select Company:", available_companies)
        
        # Get available periods
        company_data = df[df['Company'] == company]
        available_years = sorted(company_data['Year'].unique(), reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.selectbox("Select Year:", available_years)
        
        with col2:
            # Check available period types for the year
            year_data = company_data[company_data['Year'] == year]
            available_periods = []
            
            if not year_data[year_data['Period_Type'] == 'Annual'].empty:
                available_periods.append("Annual")
            
            quarterly_data = year_data[year_data['Period_Type'] == 'Quarterly']
            for q in sorted(quarterly_data['Quarter'].unique()):
                if q > 0:
                    available_periods.append(f"Q{int(q)}")
            
            if available_periods:
                period = st.selectbox("Select Period:", available_periods)
            else:
                st.error(f"No data available for {company} in {year}")
                st.stop()
        
        # Get selected data
        if period == "Annual":
            selected_data = year_data[year_data['Period_Type'] == 'Annual'].iloc[0]
        else:
            quarter_num = int(period[1])
            selected_data = year_data[year_data['Quarter'] == quarter_num].iloc[0]
        
        # Display analysis
        st.subheader(f"📈 {company} - {year} {period}")
        
        # Financial metrics display - FIXED: All show as percentages
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 💰 Profitability")
            if pd.notna(selected_data.get('ROE')):
                st.metric("ROE", f"{selected_data['ROE']:.1f}%")
            if pd.notna(selected_data.get('ROA')):
                st.metric("ROA", f"{selected_data['ROA']:.1f}%")
            if pd.notna(selected_data.get('Net Profit Margin')):
                st.metric("Net Profit Margin", f"{selected_data['Net Profit Margin']:.1f}%")
        
        with col2:
            st.markdown("#### ⚖️ Financial Health")
            if pd.notna(selected_data.get('Current Ratio')):
                st.metric("Current Ratio", f"{selected_data['Current Ratio']:.2f}")
            if pd.notna(selected_data.get('Debt-to-Equity')):
                st.metric("Debt-to-Equity", f"{selected_data['Debt-to-Equity']:.2f}")
        
        with col3:
            st.markdown("#### 📊 Efficiency")
            if pd.notna(selected_data.get('Gross Margin')):
                st.metric("Gross Margin", f"{selected_data['Gross Margin']:.1f}%")  # FIXED
            if pd.notna(selected_data.get('Debt-to-Assets')):
                st.metric("Debt-to-Assets", f"{selected_data['Debt-to-Assets']:.1f}%")  # FIXED
        
        # AI Analysis Button
        if st.button("🤖 Generate AI Analysis", type="primary", key="company_analysis"):
            with st.spinner("Analyzing financial data..."):
                analysis_data = selected_data.to_dict()
                results = financial_ai.comprehensive_analysis(analysis_data)
                
                st.markdown("---")
                st.subheader("🎯 AI Investment Analysis")
                
                # Display results
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("AI Predicted ROE", f"{results['predicted_roe']:.1f}%")  # FIXED
                
                with col_b:
                    rec = results['investment_recommendation']
                    if rec in ["Strong Buy", "Buy"]:
                        st.success(f"📈 {rec}")
                    elif "Hold" in rec:
                        st.warning(f"⚖️ {rec}")
                    else:
                        st.error(f"📉 {rec}")
                
                with col_c:
                    confidence = results['investment_confidence']
                    st.metric("AI Confidence", f"{confidence:.0%}")
                
                # Investment score and status
                score = results['investment_score']
                status = results['company_status']
                
                col_d, col_e = st.columns(2)
                
                with col_d:
                    st.metric("Investment Score", f"{score}/100")
                    st.progress(score / 100)
                
                with col_e:
                    if status == "Excellent":
                        st.success(f"🌟 Company Status: {status}")
                    elif status == "Good":
                        st.info(f"👍 Company Status: {status}")
                    elif status == "Average":
                        st.warning(f"📊 Company Status: {status}")
                    else:
                        st.error(f"⚠️ Company Status: {status}")

# Custom Analysis Page  
elif page == "🎯 Custom Analysis":
    st.header("🎯 Custom Financial Analysis")
    st.markdown("*Input your own financial ratios for AI analysis*")
    
    # Custom input form
    st.subheader("📝 Enter Financial Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Company Information")
        custom_company = st.selectbox("Company Type:", ["Almarai", "Savola", "NADEC", "Custom Company"])
        custom_year = st.number_input("Year:", min_value=2020, max_value=2030, value=2024)
        custom_quarter = st.selectbox("Period:", ["Annual", "Q1", "Q2", "Q3", "Q4"])
        
        st.markdown("#### Profitability Ratios")
        gross_margin = st.slider("Gross Margin (%)", 0, 100, 30)  # Keep as percentage input
        net_profit_margin = st.slider("Net Profit Margin (%)", 0, 50, 10)  # Keep as percentage input
        roa = st.slider("Return on Assets (%)", 0, 30, 8)  # Keep as percentage input
    
    with col2:
        st.markdown("#### Financial Health Ratios")
        current_ratio = st.slider("Current Ratio", 0.0, 5.0, 1.5, 0.1)
        debt_to_equity = st.slider("Debt-to-Equity", 0.0, 5.0, 0.8, 0.1)
        debt_to_assets = st.slider("Debt-to-Assets (%)", 0, 100, 45)  # Keep as percentage input
        
        st.markdown("#### Optional")
        manual_roe = st.slider("Manual ROE (%) - Optional", 0, 50, 12)  # Keep as percentage input
        use_manual_roe = st.checkbox("Use Manual ROE (skip AI prediction)")
    
    # Analysis button
    if st.button("🔍 ANALYZE CUSTOM DATA", type="primary", key="custom_analysis"):
        # Prepare custom data - keep values as entered (already percentages)
        custom_data = {
            'Company': custom_company,
            'Year': custom_year,
            'Quarter': 0 if custom_quarter == "Annual" else int(custom_quarter[1]),
            'Gross Margin': gross_margin,
            'Net Profit Margin': net_profit_margin,
            'ROA': roa,
            'Current Ratio': current_ratio,
            'Debt-to-Equity': debt_to_equity,
            'Debt-to-Assets': debt_to_assets
        }
        
        # Add manual ROE if specified
        if use_manual_roe:
            custom_data['ROE'] = manual_roe
        
        with st.spinner("🤖 AI is analyzing your data..."):
            # Get AI analysis
            results = financial_ai.comprehensive_analysis(custom_data)
            
            st.markdown("---")
            st.subheader(f"🎯 Analysis Results: {custom_company}")
            
            # Create results display
            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
            
            with result_col1:
                roe_display = manual_roe if use_manual_roe else results['predicted_roe']
                st.metric("ROE", f"{roe_display:.1f}%", help="Return on Equity")  # FIXED
            
            with result_col2:
                rec = results['investment_recommendation']
                color = "🟢" if rec in ["Strong Buy", "Buy"] else "🟡" if "Hold" in rec else "🔴"
                st.metric("Investment Rec", f"{color} {rec}")
            
            with result_col3:
                st.metric("AI Confidence", f"{results['investment_confidence']:.0%}")
            
            with result_col4:
                st.metric("Investment Score", f"{results['investment_score']}/100")
            
            # Detailed analysis
            st.markdown("#### 📊 Detailed Analysis")
            
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                st.markdown("**✅ Strengths:**")
                strengths = []
                if roa > 8:
                    strengths.append(f"Strong ROA ({roa:.1f}%)")  # FIXED
                if net_profit_margin > 10:
                    strengths.append(f"Good profitability ({net_profit_margin:.1f}%)")  # FIXED
                if current_ratio > 1.5:
                    strengths.append(f"Strong liquidity ({current_ratio:.2f})")
                if debt_to_equity < 0.8:
                    strengths.append(f"Conservative debt ({debt_to_equity:.2f})")
                
                if strengths:
                    for strength in strengths:
                        st.write(f"• {strength}")
                else:
                    st.write("• Focus on improving key metrics")
            
            with analysis_col2:
                st.markdown("**⚠️ Areas for Improvement:**")
                concerns = []
                if roa < 5:
                    concerns.append(f"Low ROA ({roa:.1f}%) - improve asset efficiency")  # FIXED
                if net_profit_margin < 5:
                    concerns.append(f"Low margins ({net_profit_margin:.1f}%) - reduce costs")  # FIXED
                if current_ratio < 1.2:
                    concerns.append(f"Liquidity risk ({current_ratio:.2f}) - improve cash flow")
                if debt_to_equity > 1.2:
                    concerns.append(f"High leverage ({debt_to_equity:.2f}) - reduce debt")
                
                if concerns:
                    for concern in concerns:
                        st.write(f"• {concern}")
                else:
                    st.write("• Strong performance across all metrics!")
            
            # Investment recommendation explanation
            st.markdown("#### 💡 Investment Recommendation Explanation")
            
            score = results['investment_score']
            if score >= 70:
                st.success("🌟 **Strong Buy Recommendation**: Outstanding financial performance with multiple strength indicators. Low risk profile with strong growth potential.")
            elif score >= 60:
                st.success("📈 **Buy Recommendation**: Good financial performance with solid fundamentals. Moderate risk with positive outlook.")
            elif score >= 40:
                st.warning("⚖️ **Hold Recommendation**: Average performance with mixed signals. Monitor for improvements before increasing position.")
            else:
                st.error("📉 **Sell Recommendation**: Below-average performance with concerning metrics. Consider reducing exposure until fundamentals improve.")

# Ratio Prediction Page
elif page == "🔮 Ratio Prediction":
    st.header("🔮 Financial Ratio Prediction")
    st.markdown("*Predict future financial performance using AI and historical trends*")
    
    if not df.empty:
        # Prediction settings
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            available_companies = sorted(df['Company'].unique())
            pred_company = st.selectbox("Company for Prediction:", available_companies)
            prediction_year = st.selectbox("Predict for Year:", [2024, 2025, 2026, 2027])
        
        with pred_col2:
            prediction_method = st.selectbox("Prediction Method:", 
                                           ["AI Model (Advanced)", "Trend Analysis", "Conservative Estimate"])
            confidence_level = st.slider("Confidence Level", 50, 95, 85)
        
        # Get historical data for context
        company_historical = df[df['Company'] == pred_company]
        annual_historical = company_historical[company_historical['Period_Type'] == 'Annual'].sort_values('Year')
        
        if len(annual_historical) >= 2:
            latest_data = annual_historical.iloc[-1]
            
            st.subheader(f"📊 Historical Context: {pred_company}")
            
            # Show recent trend
            recent_years = annual_historical.tail(3)
            
            trend_col1, trend_col2, trend_col3 = st.columns(3)
            
            for i, (_, year_data) in enumerate(recent_years.iterrows()):
                with [trend_col1, trend_col2, trend_col3][i]:
                    st.markdown(f"**{int(year_data['Year'])}**")
                    if pd.notna(year_data.get('ROE')):
                        st.metric("ROE", f"{year_data['ROE']:.1f}%")  # FIXED
                    if pd.notna(year_data.get('ROA')):
                        st.metric("ROA", f"{year_data['ROA']:.1f}%")  # FIXED
            
            # Prediction button
            if st.button("🎯 Generate Predictions", type="primary"):
                with st.spinner("🔮 Generating predictions..."):
                    # Create prediction data based on latest
                    prediction_data = latest_data.to_dict()
                    prediction_data['Year'] = prediction_year
                    
                    # Get AI prediction
                    if prediction_method == "AI Model (Advanced)":
                        results = financial_ai.comprehensive_analysis(prediction_data)
                        predicted_roe = results['predicted_roe']
                        investment_rec = results['investment_recommendation']
                        confidence = results['investment_confidence']
                    else:
                        # Simple trend-based prediction
                        if len(annual_historical) >= 2:
                            recent_roe = annual_historical['ROE'].tail(2).values
                            if len(recent_roe) == 2 and not np.isnan(recent_roe).any():
                                growth_rate = (recent_roe[1] - recent_roe[0]) / recent_roe[0] if recent_roe[0] != 0 else 0
                                
                                if prediction_method == "Conservative Estimate":
                                    growth_rate *= 0.5  # More conservative
                                
                                predicted_roe = recent_roe[1] * (1 + growth_rate)
                                predicted_roe = max(0, min(predicted_roe, 50))  # Bound predictions (percentage format)
                            else:
                                predicted_roe = latest_data.get('ROE', 10.0)
                        else:
                            predicted_roe = latest_data.get('ROE', 10.0)
                        
                        # Simple recommendation based on predicted ROE
                        if predicted_roe > 15:
                            investment_rec = "Buy"
                            confidence = 0.75
                        elif predicted_roe > 8:
                            investment_rec = "Hold"
                            confidence = 0.70
                        else:
                            investment_rec = "Sell"
                            confidence = 0.65
                    
                    st.markdown("---")
                    st.subheader(f"🔮 {pred_company} - {prediction_year} Predictions")
                    
                    # Display predictions
                    pred_result_col1, pred_result_col2, pred_result_col3 = st.columns(3)
                    
                    with pred_result_col1:
                        current_roe = latest_data.get('ROE', 0)
                        roe_change = ((predicted_roe - current_roe) / current_roe * 100) if current_roe != 0 else 0
                        st.metric("Predicted ROE", f"{predicted_roe:.1f}%", f"{roe_change:+.1f}%")  # FIXED
                    
                    with pred_result_col2:
                        st.metric("Investment Outlook", investment_rec)
                    
                    with pred_result_col3:
                        st.metric("Prediction Confidence", f"{confidence:.0%}")
                    
                    # Prediction chart
                    st.subheader("📈 ROE Trend & Prediction")
                    
                    # Prepare chart data
                    chart_data = []
                    
                    # Historical data
                    for _, row in annual_historical.iterrows():
                        if pd.notna(row.get('ROE')):
                            chart_data.append({
                                'Year': int(row['Year']),
                                'ROE': row['ROE'],
                                'Type': 'Historical'
                            })
                    
                    # Predicted data
                    chart_data.append({
                        'Year': prediction_year,
                        'ROE': predicted_roe,
                        'Type': 'Predicted'
                    })
                    
                    chart_df = pd.DataFrame(chart_data)
                    
                    # Create line chart
                    fig = px.line(chart_df, x='Year', y='ROE', color='Type',
                                 title=f"{pred_company} ROE: Historical vs Predicted",
                                 markers=True)
                    fig.update_layout(yaxis_title="ROE (%)")  # FIXED
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk assessment
                    st.subheader("⚠️ Prediction Risk Assessment")
                    
                    risk_factors = []
                    if confidence < 0.7:
                        risk_factors.append("Low prediction confidence due to limited historical data")
                    if abs(roe_change) > 20:
                        risk_factors.append("High volatility predicted - monitor closely")
                    if len(annual_historical) < 3:
                        risk_factors.append("Limited historical data available for trend analysis")
                    
                    if risk_factors:
                        for risk in risk_factors:
                            st.warning(f"⚠️ {risk}")
                    else:
                        st.success("✅ Prediction based on stable historical patterns")
        
        else:
            st.warning(f"⚠️ Insufficient historical data for {pred_company}. Need at least 2 years of data for predictions.")

# Health Check Page
elif page == "🏥 Health Check":
    st.header("🏥 Financial Health Assessment")
    st.markdown("*Comprehensive health analysis using multiple financial indicators*")
    
    if not df.empty:
        # Health check settings
        health_company = st.selectbox("Select Company for Health Check:", sorted(df['Company'].unique()))
        
        # Get latest data
        company_data = df[df['Company'] == health_company]
        annual_data = company_data[company_data['Period_Type'] == 'Annual'].sort_values('Year')
        
        if not annual_data.empty:
            latest_data = annual_data.iloc[-1]
            
            # Health assessment
            if st.button("🔍 Perform Health Check", type="primary"):
                with st.spinner("🏥 Analyzing financial health..."):
                    # Get AI analysis
                    results = financial_ai.comprehensive_analysis(latest_data.to_dict())
                    
                    st.markdown("---")
                    st.subheader(f"🏥 Health Report: {health_company}")
                    st.markdown(f"*Assessment Period: {int(latest_data['Year'])} Annual*")
                    
                    # Overall health score
                    health_score = results['investment_score']
                    
                    health_col1, health_col2 = st.columns([1, 2])
                    
                    with health_col1:
                        st.metric("Overall Health Score", f"{health_score}/100")
                        st.progress(health_score / 100)
                        
                        # Health grade
                        if health_score >= 80:
                            st.success("🌟 Grade: A (Excellent)")
                        elif health_score >= 70:
                            st.success("👍 Grade: B (Good)")
                        elif health_score >= 60:
                            st.info("📊 Grade: C (Average)")
                        elif health_score >= 50:
                            st.warning("⚠️ Grade: D (Below Average)")
                        else:
                            st.error("🚨 Grade: F (Poor)")
                    
                    with health_col2:
                        st.markdown("#### 📊 Health Indicators")
                        
                        # Individual health checks - UPDATED for percentage values
                        health_indicators = [
                            ("Profitability", latest_data.get('ROE', 0), 12, "ROE"),  # Updated threshold
                            ("Asset Efficiency", latest_data.get('ROA', 0), 8, "ROA"),  # Updated threshold
                            ("Profit Margins", latest_data.get('Net Profit Margin', 0), 10, "NPM"),  # Updated threshold
                            ("Liquidity", latest_data.get('Current Ratio', 0), 1.2, "CR"),
                            ("Leverage", latest_data.get('Debt-to-Equity', 0), 1.0, "D/E", True)  # Lower is better
                        ]
                        
                        for indicator, value, benchmark, code, *lower_better in health_indicators:
                            is_lower_better = len(lower_better) > 0 and lower_better[0]
                            
                            if pd.notna(value):
                                if is_lower_better:
                                    status = "✅ Healthy" if value <= benchmark else "⚠️ Risk" if value <= benchmark * 1.5 else "🚨 High Risk"
                                else:
                                    status = "✅ Healthy" if value >= benchmark else "⚠️ Below Par" if value >= benchmark * 0.7 else "🚨 Poor"
                                
                                if code in ["ROE", "ROA", "NPM"]:
                                    value_str = f"{value:.1f}%"  # FIXED
                                else:
                                    value_str = f"{value:.2f}"
                                
                                st.write(f"**{indicator}:** {value_str} {status}")
                            else:
                                st.write(f"**{indicator}:** Data not available")
                    
                    # Trend analysis
                    if len(annual_data) >= 3:
                        st.subheader("📈 Health Trend Analysis")
                        
                        # Calculate trends for key metrics
                        recent_data = annual_data.tail(3)
                        
                        trend_metrics = ['ROE', 'ROA', 'Net Profit Margin']
                        trend_cols = st.columns(len(trend_metrics))
                        
                        for i, metric in enumerate(trend_metrics):
                            with trend_cols[i]:
                                if metric in recent_data.columns:
                                    values = recent_data[metric].values
                                    if len(values) >= 2 and not np.isnan(values).all():
                                        # Simple trend calculation
                                        trend = "📈 Improving" if values[-1] > values[-2] else "📉 Declining" if values[-1] < values[-2] else "➡️ Stable"
                                        
                                        st.metric(
                                            f"{metric} Trend",
                                            trend,
                                            f"{values[-1]:.1f}%" if not np.isnan(values[-1]) else "N/A"  # FIXED
                                        )
                    
                    # Recommendations - UPDATED for percentage values
                    st.subheader("💡 Health Improvement Recommendations")
                    
                    recommendations = []
                    
                    if latest_data.get('ROE', 0) < 10:  # Updated for percentage
                        recommendations.append("🎯 Focus on improving return on equity through better profit margins or asset efficiency")
                    
                    if latest_data.get('Current Ratio', 0) < 1.2:
                        recommendations.append("💧 Improve liquidity by increasing current assets or reducing short-term liabilities")
                    
                    if latest_data.get('Debt-to-Equity', 0) > 1.0:
                        recommendations.append("⚖️ Consider debt reduction to improve financial stability")
                    
                    if latest_data.get('Net Profit Margin', 0) < 8:  # Updated for percentage
                        recommendations.append("💰 Work on cost optimization to improve profit margins")
                    
                    if not recommendations:
                        recommendations.append("🌟 Excellent financial health! Continue current management strategies")
                    
                    for rec in recommendations:
                        st.write(f"• {rec}")
        
        else:
            st.error(f"❌ No annual data available for {health_company}")

# Comparison Page
elif page == "⚖️ Comparison":
    st.header("⚖️ Company Comparison Analysis")
    st.markdown("*Side-by-side financial performance comparison*")
    
    if not df.empty:
        # Comparison settings
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            available_years = sorted(df['Year'].unique(), reverse=True)
            comp_year = st.selectbox("Comparison Year:", available_years)
        
        with comp_col2:
            # Check available periods for the year
            year_data = df[df['Year'] == comp_year]
            available_periods = ["Annual"]
            
            quarterly_data = year_data[year_data['Period_Type'] == 'Quarterly']
            for q in sorted(quarterly_data['Quarter'].unique()):
                if q > 0:
                    available_periods.append(f"Q{int(q)}")
            
            comp_period = st.selectbox("Period:", available_periods)
        
        # Get comparison data
        if comp_period == "Annual":
            comparison_data = df[(df['Year'] == comp_year) & (df['Period_Type'] == 'Annual')]
        else:
            quarter_num = int(comp_period[1])
            comparison_data = df[(df['Year'] == comp_year) & (df['Quarter'] == quarter_num)]
        
        if not comparison_data.empty:
            st.subheader(f"📊 Company Comparison - {comp_year} {comp_period}")
            
            # Create comparison table
            metrics_to_compare = ['ROE', 'ROA', 'Net Profit Margin', 'Current Ratio', 'Debt-to-Equity']
            available_metrics = [m for m in metrics_to_compare if m in comparison_data.columns]
            
            if available_metrics:
                # Prepare display data - FIXED: Already in percentage format
                display_data = comparison_data[['Company'] + available_metrics].copy()
                
                # Format for display
                for metric in available_metrics:
                    if metric in ['ROE', 'ROA', 'Net Profit Margin']:
                        display_data[f"{metric} (%)"] = display_data[metric].round(1)  # Already in percentage
                        display_data = display_data.drop(columns=[metric])
                    else:
                        display_data[metric] = display_data[metric].round(2)
                
                # Display comparison table
                st.dataframe(display_data.set_index('Company'), use_container_width=True)
                
                # Comparison charts
                st.subheader("📈 Visual Comparison")
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # ROE comparison
                    if 'ROE' in comparison_data.columns:
                        roe_data = comparison_data[['Company', 'ROE']].copy()
                        fig_roe = px.bar(roe_data, x='Company', y='ROE',
                                        title=f"ROE Comparison - {comp_year} {comp_period}",
                                        color='ROE', color_continuous_scale='viridis')
                        fig_roe.update_layout(yaxis_title="ROE (%)", showlegend=False)  # FIXED
                        st.plotly_chart(fig_roe, use_container_width=True)
                
                with chart_col2:
                    # Current Ratio comparison
                    if 'Current Ratio' in comparison_data.columns:
                        cr_data = comparison_data[['Company', 'Current Ratio']].copy()
                        fig_cr = px.bar(cr_data, x='Company', y='Current Ratio',
                                       title=f"Liquidity Comparison - {comp_year} {comp_period}",
                                       color='Current Ratio', color_continuous_scale='plasma')
                        fig_cr.update_layout(showlegend=False)
                        st.plotly_chart(fig_cr, use_container_width=True)
                
                # Performance ranking
                st.subheader("🏆 Performance Ranking")
                
                # Calculate overall scores for ranking
                ranking_data = []
                
                for _, company_row in comparison_data.iterrows():
                    company_dict = company_row.to_dict()
                    results = financial_ai.comprehensive_analysis(company_dict)
                    
                    ranking_data.append({
                        'Company': company_row['Company'],
                        'Overall Score': results['investment_score'],
                        'Investment Rec': results['investment_recommendation'],
                        'AI Confidence': f"{results['investment_confidence']:.0%}"
                    })
                
                ranking_df = pd.DataFrame(ranking_data).sort_values('Overall Score', ascending=False)
                
                # Display ranking
                rank_cols = st.columns(len(ranking_df))
                
                for i, (_, company_data) in enumerate(ranking_df.iterrows()):
                    with rank_cols[i]:
                        position = i + 1
                        medal = "🥇" if position == 1 else "🥈" if position == 2 else "🥉" if position == 3 else f"{position}."
                        
                        st.markdown(f"### {medal} {company_data['Company']}")
                        st.metric("Score", f"{company_data['Overall Score']}/100")
                        
                        rec = company_data['Investment Rec']
                        if rec in ["Strong Buy", "Buy"]:
                            st.success(f"📈 {rec}")
                        elif "Hold" in rec:
                            st.warning(f"⚖️ {rec}")
                        else:
                            st.error(f"📉 {rec}")
                        
                        st.write(f"Confidence: {company_data['AI Confidence']}")
            
            else:
                st.warning("⚠️ No comparable metrics available for the selected period")
        
        else:
            st.error(f"❌ No data available for {comp_year} {comp_period}")

# Footer with information
st.markdown("---")
st.markdown("### 🤖 Financial AI Assistant Information")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("**📊 Data Coverage**")
    if not df.empty:
        st.write(f"• Period: {df['Year'].min()}-{df['Year'].max()}")
        st.write(f"• Companies: {df['Company'].nunique()}")
        st.write(f"• Records: {len(df)}")
    else:
        st.write("• No data loaded")

with info_col2:
    st.markdown("**🤖 AI Models**")
    models_loaded = sum(1 for status in model_status.values() if "✅" in status)
    total_models = len(model_status)
    st.write(f"• Models: {models_loaded}/{total_models} loaded")
    st.write(f"• Prediction: {'✅ Available' if models['roe_model'] else '⚠️ Fallback'}")
    st.write(f"• Classification: {'✅ Available' if models['investment_model'] else '⚠️ Fallback'}")

with info_col3:
    st.markdown("**📈 Capabilities**")
    st.write("• ROE Prediction")
    st.write("• Investment Recommendations")
    st.write("• Financial Health Assessment")
    st.write("• Company Comparison")
    st.write("• Custom Analysis")

# Model status expander
with st.expander("🔧 Technical Details", expanded=False):
    st.markdown("**Model Status:**")
    for model_name, status in model_status.items():
        st.write(f"• {model_name}: {status}")
    
    st.markdown("**Features:**")
    st.write("• Real-time financial analysis")
    st.write("• Machine learning predictions")
    st.write("• Interactive visualizations")
    st.write("• Historical trend analysis")
    st.write("• Risk assessment")

st.markdown("---")
st.markdown("*Saudi Food Sector Financial AI Assistant | Powered by Advanced Machine Learning*")
