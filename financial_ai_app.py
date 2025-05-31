import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import os
import statsmodels.api as sm
warnings.filterwarnings('ignore')

# Import comparison components
from comparison_components import company_comparison_component, industry_analysis_component

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

# Function to load financial data
def load_financial_data():
    """
    Load financial data from CSV file or create sample data if file not found.
    """
    try:
        # Check if uploaded file exists in session state
        if 'financial_data' in st.session_state:
            return st.session_state.financial_data
        
        # Try to load from default path
        file_path = "/home/ubuntu/upload/SavolaAlmaraiNADECFinancialRatiosCSV.csv.csv"
        if os.path.exists(file_path):
            # Initialize DataLoader
            data_dir = os.path.join(os.getcwd(), 'data')
            loader = DataLoader(data_dir)
            
            # Load data
            success = loader.load_csv(file_path)
            
            if success and loader.processed_data is not None:
                st.success(f"✅ Data loaded from: {os.path.basename(file_path)}")
                
                # Store in session state
                st.session_state.financial_data = loader.processed_data
                return loader.processed_data
        
        # If we get here, data loading failed
        st.error("Failed to load financial data. Please upload a CSV file.")
        return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Error loading financial data: {e}")
        return pd.DataFrame()

# Function to load AI models
def load_ai_models():
    """
    Load pre-trained AI models for financial analysis.
    """
    try:
        # Check if models exist in session state
        if 'models' in st.session_state and 'encoders' in st.session_state:
            return st.session_state.models, st.session_state.encoders, True
        
        # Initialize empty models and encoders
        models = {
            'recommendation': None,
            'roe_prediction': None
        }
        
        encoders = {
            'company': None,
            'period': None
        }
        
        # Try to load models from default path
        model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Check if models exist
        model_files = {
            'recommendation': os.path.join(model_dir, 'recommendation_model.joblib'),
            'roe_prediction': os.path.join(model_dir, 'roe_prediction_model.joblib')
        }
        
        encoder_files = {
            'company': os.path.join(model_dir, 'company_encoder.joblib'),
            'period': os.path.join(model_dir, 'period_encoder.joblib')
        }
        
        # Load models if they exist
        for model_name, model_path in model_files.items():
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
        
        # Load encoders if they exist
        for encoder_name, encoder_path in encoder_files.items():
            if os.path.exists(encoder_path):
                encoders[encoder_name] = joblib.load(encoder_path)
        
        # Store in session state
        st.session_state.models = models
        st.session_state.encoders = encoders
        
        # Check if all models and encoders are loaded
        all_loaded = all(model is not None for model in models.values()) and all(encoder is not None for encoder in encoders.values())
        
        return models, encoders, all_loaded
    
    except Exception as e:
        st.error(f"Error loading AI models: {e}")
        return {}, {}, False

# DataLoader class for robust data handling
class DataLoader:
    """
    A flexible data loader for financial ratio data that:
    1. Loads data from CSV files (specifically designed for Savola, Almarai, and NADEC)
    2. Supports adding new companies easily
    3. Preprocesses data for analysis
    4. Provides consistent data access for all system components
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the DataLoader with data directory path.
        
        Parameters:
        -----------
        data_dir : str
            Path to the directory containing financial data
        """
        self.data_dir = data_dir
        if data_dir:
            self.raw_dir = os.path.join(data_dir, 'raw')
            self.processed_dir = os.path.join(data_dir, 'processed')
            
            # Create directories if they don't exist
            os.makedirs(self.raw_dir, exist_ok=True)
            os.makedirs(self.processed_dir, exist_ok=True)
        
        # Initialize data containers
        self.raw_data = None
        self.processed_data = None
        self.companies = []
        self.available_ratios = []
        
        st.write("DataLoader initialized successfully")
    
    def load_csv(self, file_path=None, data=None, copy_to_raw=False):
        """
        Load financial ratio data from a CSV file or DataFrame.
        
        Parameters:
        -----------
        file_path : str, optional
            Path to the CSV file
        data : pandas.DataFrame, optional
            DataFrame containing financial data
        copy_to_raw : bool, optional
            Whether to copy the file to the raw data directory
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Check if file exists or DataFrame is provided
            if data is not None:
                st.write("Loading data from provided DataFrame...")
                self.raw_data = data.copy()
            elif file_path:
                if not os.path.exists(file_path):
                    st.warning(f"File not found: {file_path}")
                    return False
                
                # Load data
                st.write(f"Loading data from {file_path}...")
                self.raw_data = pd.read_csv(file_path)
                
                # Copy file to raw directory if requested
                if copy_to_raw and self.data_dir:
                    import shutil
                    raw_file_path = os.path.join(self.raw_dir, os.path.basename(file_path))
                    shutil.copy2(file_path, raw_file_path)
                    st.write(f"File copied to {raw_file_path}")
            else:
                st.warning("No data source provided")
                return False
            
            # Extract companies and available ratios
            if 'Company' in self.raw_data.columns:
                self.companies = sorted(self.raw_data['Company'].unique().tolist())
                st.write(f"Found {len(self.companies)} companies: {', '.join(self.companies)}")
            
            # Process data
            self._process_data()
            
            return True
        
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def _process_data(self):
        """
        Process raw data to prepare it for analysis.
        Specifically handles Savola, Almarai, and NADEC financial data format.
        """
        if self.raw_data is None or self.raw_data.empty:
            st.warning("No raw data available for processing")
            return
        
        try:
            st.write("Processing data...")
            processed_data = self.raw_data.copy()
            
            # Clean column names - strip whitespace and remove unnamed columns
            processed_data.columns = [col.strip() if isinstance(col, str) else col for col in processed_data.columns]
            processed_data = processed_data.loc[:, ~processed_data.columns.str.contains('^Unnamed')]
            
            # Ensure required columns exist
            required_columns = ['Company', 'Period', 'Period_Type']
            missing_columns = [col for col in required_columns if col not in processed_data.columns]
            
            if missing_columns:
                st.warning(f"Missing required columns: {', '.join(missing_columns)}")
                
                # Try to infer missing columns
                if 'Company' not in processed_data.columns and 'company' in processed_data.columns:
                    processed_data['Company'] = processed_data['company']
                
                if 'Period' not in processed_data.columns:
                    if 'Year' in processed_data.columns and 'Quarter' in processed_data.columns:
                        processed_data['Period'] = processed_data.apply(
                            lambda row: f"{int(float(row['Year']))}-Q{int(float(row['Quarter']))}" 
                            if pd.notna(row['Year']) and pd.notna(row['Quarter']) and row['Quarter'] != 0 
                            else f"{int(float(row['Year']))} Annual", axis=1)
                    elif 'Year' in processed_data.columns:
                        processed_data['Period'] = processed_data['Year'].astype(str)
                
                if 'Period_Type' not in processed_data.columns:
                    if 'Quarter' in processed_data.columns:
                        # Mark records with Quarter=0 as Annual, others as Quarterly
                        processed_data['Period_Type'] = processed_data['Quarter'].apply(
                            lambda q: 'Annual' if pd.notna(q) and float(q) == 0 else 'Quarterly'
                        )
                    else:
                        processed_data['Period_Type'] = 'Annual'
            
            # Convert date columns if they exist
            date_columns = ['Date', 'date']
            for col in date_columns:
                if col in processed_data.columns:
                    processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
                    
                    # Rename to standardized column name
                    if col != 'Date':
                        processed_data['Date'] = processed_data[col]
                        processed_data = processed_data.drop(columns=[col])
            
            # If Date column doesn't exist, try to create it
            if 'Date' not in processed_data.columns:
                if 'Year' in processed_data.columns and 'Quarter' in processed_data.columns:
                    # Create approximate dates for quarters with robust error handling
                    def quarter_to_date(row):
                        try:
                            if pd.isna(row['Year']) or pd.isna(row['Quarter']):
                                return None
                            
                            year = int(float(row['Year']))
                            quarter = int(float(row['Quarter']))
                            
                            # Handle annual data (Quarter=0) differently
                            if quarter == 0:
                                # For annual data, use January 1st of the year
                                return pd.Timestamp(year=year, month=1, day=1)
                            elif not (1 <= quarter <= 4):
                                st.warning(f"Invalid quarter value {quarter}, using 1")
                                quarter = 1
                                
                            month = (quarter - 1) * 3 + 1  # Q1->1, Q2->4, Q3->7, Q4->10
                            return pd.Timestamp(year=year, month=month, day=1)
                        except Exception as e:
                            st.warning(f"Could not convert Year/Quarter to date: {e}")
                            return None
                    
                    processed_data['Date'] = processed_data.apply(quarter_to_date, axis=1)
                
                elif 'Year' in processed_data.columns:
                    # Create dates for years (January 1st) with error handling
                    processed_data['Date'] = pd.to_datetime(processed_data['Year'].astype(str) + '-01-01', errors='coerce')
                
                # If Period column contains dates (like '3/31/2016'), try to parse them
                elif 'Period' in processed_data.columns:
                    try:
                        processed_data['Date'] = pd.to_datetime(processed_data['Period'], errors='coerce')
                    except:
                        st.warning("Could not convert Period to Date")
            
            # Clean financial ratio columns
            financial_columns = ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE', 
                               'Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets']
            
            for col in financial_columns:
                if col in processed_data.columns:
                    # If column is string type, clean it
                    if processed_data[col].dtype == 'object':
                        # Remove % signs, commas, etc.
                        processed_data[col] = processed_data[col].astype(str)
                        processed_data[col] = processed_data[col].str.replace('%', '', regex=False)
                        processed_data[col] = processed_data[col].str.replace(',', '', regex=False)
                        processed_data[col] = processed_data[col].str.strip()
                        
                    # Convert to numeric
                    processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
                    
                    # Check if values are percentages (>1) and need conversion to decimal
                    if col in ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE'] and processed_data[col].max() > 1:
                        # Check if values look like percentages (e.g., 15.7 for 15.7%)
                        if processed_data[col].max() <= 100:
                            processed_data[col] = processed_data[col] / 100
                            st.write(f"Converted {col} from percentage to decimal format")
            
            # Extract available ratios (excluding metadata columns)
            metadata_columns = ['Period', 'Period_Type', 'Year', 'Quarter', 'Company', 'Date']
            self.available_ratios = [col for col in processed_data.columns 
                                   if col not in metadata_columns]
            
            st.write(f"Found {len(self.available_ratios)} financial ratios")
            
            # Handle missing values
            for ratio in self.available_ratios:
                missing_count = processed_data[ratio].isna().sum()
                if missing_count > 0:
                    st.warning(f"{missing_count} missing values in {ratio}")
                    
                    # Fill missing values with median per company
                    for company in processed_data['Company'].unique():
                        company_median = processed_data[processed_data['Company'] == company][ratio].median()
                        if not pd.isna(company_median):
                            mask = (processed_data['Company'] == company) & (processed_data[ratio].isna())
                            processed_data.loc[mask, ratio] = company_median
                            st.write(f"  - Filled {mask.sum()} missing values for {company} with median {company_median:.4f}")
            
            # Store processed data
            self.processed_data = processed_data
            
            # Save processed data if directory exists
            if self.data_dir:
                self._save_processed_data()
            
            st.success("Data processing completed successfully")
        
        except Exception as e:
            st.error(f"Error processing data: {e}")
    
    def _save_processed_data(self):
        """
        Save processed data to CSV file.
        """
        if self.processed_data is None or self.processed_data.empty:
            st.warning("No processed data available to save")
            return
        
        try:
            # Create file path
            file_path = os.path.join(self.processed_dir, 'processed_financial_ratios.csv')
            
            # Save to CSV
            self.processed_data.to_csv(file_path, index=False)
            
            st.write(f"Processed data saved to {file_path}")
        
        except Exception as e:
            st.error(f"Error saving processed data: {e}")
    
    def add_company_data(self, company_data, company_name=None):
        """
        Add data for a new company.
        
        Parameters:
        -----------
        company_data : pandas.DataFrame or str
            DataFrame containing company data or path to CSV file
        company_name : str, optional
            Name of the company (required if not in the data)
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Load data if file path is provided
            if isinstance(company_data, str):
                if not os.path.exists(company_data):
                    st.warning(f"File not found: {company_data}")
                    return False
                
                company_data = pd.read_csv(company_data)
            
            # Ensure company_data is a DataFrame
            if not isinstance(company_data, pd.DataFrame):
                st.warning("Company data must be a DataFrame or path to CSV file")
                return False
            
            # Check if company name is provided or in the data
            if 'Company' not in company_data.columns and company_name is None:
                st.warning("Company name must be provided or included in the data")
                return False
            
            # Add company name if not in the data
            if 'Company' not in company_data.columns:
                company_data['Company'] = company_name
            elif company_name is not None:
                # Override company name if provided
                company_data['Company'] = company_name
            
            # Get company name from data
            company_name = company_data['Company'].iloc[0]
            
            # Check if company already exists
            if self.processed_data is not None and company_name in self.processed_data['Company'].unique():
                st.warning(f"Company {company_name} already exists. Data will be merged.")
                
                # Remove existing company data
                self.processed_data = self.processed_data[self.processed_data['Company'] != company_name]
            
            # Process new company data
            loader = DataLoader()
            loader.load_csv(data=company_data)
            
            if loader.processed_data is None or loader.processed_data.empty:
                st.warning(f"Failed to process data for {company_name}")
                return False
            
            # Merge with existing data
            if self.processed_data is not None:
                self.processed_data = pd.concat([self.processed_data, loader.processed_data], ignore_index=True)
            else:
                self.processed_data = loader.processed_data
            
            # Update companies list
            self.companies = sorted(self.processed_data['Company'].unique().tolist())
            
            # Save processed data if directory exists
            if self.data_dir:
                self._save_processed_data()
            
            st.success(f"Data for {company_name} added successfully")
            return True
        
        except Exception as e:
            st.error(f"Error adding company data: {e}")
            return False

# Financial AI class for analysis and predictions
class FinancialAI:
    """
    A class for financial analysis and predictions using AI models.
    """
    
    def __init__(self, models=None, encoders=None):
        """
        Initialize the FinancialAI with models and encoders.
        
        Parameters:
        -----------
        models : dict, optional
            Dictionary of trained models
        encoders : dict, optional
            Dictionary of encoders for categorical variables
        """
        self.models = models or {}
        self.encoders = encoders or {}
    
    def build_portfolio(self, df, criteria):
        """
        Build an investment portfolio based on specified criteria.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing financial data
        criteria : dict
            Dictionary of criteria for portfolio selection
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing selected companies
        """
        # Create a copy of the DataFrame
        df_portfolio = df.copy()
        
        # Filter by period type (use most recent Annual data)
        if 'Period_Type' in df_portfolio.columns and 'Date' in df_portfolio.columns:
            # Filter for Annual data first
            annual_data = df_portfolio[df_portfolio['Period_Type'] == 'Annual'].copy()
            if not annual_data.empty:
                # Get the most recent date for Annual data for each company
                most_recent_annual = annual_data.groupby('Company')['Date'].max().reset_index()
                # Merge to get only the most recent Annual data for each company
                df_portfolio = annual_data.merge(most_recent_annual, on=['Company', 'Date'])
                st.info("Portfolio built using the latest Annual data for each company.") # Add feedback
            else:
                st.warning("No Annual data found. Using the absolute latest data point as fallback.")
                # Fallback: use the absolute most recent data if no annual data exists
                most_recent = df_portfolio.groupby('Company')['Date'].max().reset_index()
                df_portfolio = df_portfolio.merge(most_recent, on=['Company', 'Date'])

        
        # Debug information
        st.write("Debug: Initial portfolio size before filtering:", len(df_portfolio))
        
        # Store original criteria for reference
        original_criteria = criteria.copy()
        
        # Apply criteria filters with flexibility
        filtered_portfolio = df_portfolio.copy()
        
        # Track which filters are being applied
        filters_applied = []
        
        if 'min_roi' in criteria:
            filtered_portfolio = filtered_portfolio[filtered_portfolio['ROE'] >= criteria['min_roi']]
            filters_applied.append(f"ROE >= {criteria['min_roi']:.1%}")
            st.write(f"Debug: After ROE filter: {len(filtered_portfolio)} companies")
        
        if 'max_risk' in criteria and len(filtered_portfolio) > 0:
            filtered_portfolio = filtered_portfolio[filtered_portfolio['Debt-to-Equity'] <= criteria['max_risk']]
            filters_applied.append(f"Debt-to-Equity <= {criteria['max_risk']:.2f}")
            st.write(f"Debug: After Risk filter: {len(filtered_portfolio)} companies")
        
        if 'min_growth' in criteria and len(filtered_portfolio) > 0:
            filtered_portfolio = filtered_portfolio[filtered_portfolio['Net Profit Margin'] >= criteria['min_growth']]
            filters_applied.append(f"Net Profit Margin >= {criteria['min_growth']:.1%}")
            st.write(f"Debug: After Growth filter: {len(filtered_portfolio)} companies")
        
        if 'industry' in criteria and len(filtered_portfolio) > 0:
            filtered_portfolio = filtered_portfolio[filtered_portfolio['Company'].str.contains(criteria['industry'], case=False)]
            filters_applied.append(f"Industry contains '{criteria['industry']}'")
            st.write(f"Debug: After Industry filter: {len(filtered_portfolio)} companies")
        
        # If no companies match the criteria, try relaxed criteria
        if len(filtered_portfolio) == 0:
            st.warning("No companies match the strict criteria. Trying with relaxed criteria...")
            
            # Reset to original data
            relaxed_portfolio = df_portfolio.copy()
            relaxed_filters = []
            
            # Apply relaxed criteria (50% of original thresholds)
            if 'min_roi' in criteria:
                relaxed_roi = criteria['min_roi'] * 0.5
                relaxed_portfolio = relaxed_portfolio[relaxed_portfolio['ROE'] >= relaxed_roi]
                relaxed_filters.append(f"ROE >= {relaxed_roi:.1%}")
            
            if 'max_risk' in criteria and len(relaxed_portfolio) > 0:
                relaxed_risk = criteria['max_risk'] * 1.5
                relaxed_portfolio = relaxed_portfolio[relaxed_portfolio['Debt-to-Equity'] <= relaxed_risk]
                relaxed_filters.append(f"Debt-to-Equity <= {relaxed_risk:.2f}")
            
            if 'min_growth' in criteria and len(relaxed_portfolio) > 0:
                relaxed_growth = criteria['min_growth'] * 0.5
                relaxed_portfolio = relaxed_portfolio[relaxed_portfolio['Net Profit Margin'] >= relaxed_growth]
                relaxed_filters.append(f"Net Profit Margin >= {relaxed_growth:.1%}")
            
            # If we found matches with relaxed criteria, use those
            if len(relaxed_portfolio) > 0:
                filtered_portfolio = relaxed_portfolio
                st.info(f"Found {len(filtered_portfolio)} companies with relaxed criteria: {', '.join(relaxed_filters)}")
            else:
                # If still no matches, just return the top companies by ROE
                st.warning("No companies match even with relaxed criteria. Showing top companies by ROE.")
                filtered_portfolio = df_portfolio.sort_values('ROE', ascending=False).head(3)
        
        # Sort by ROE (descending) as default ranking
        df_portfolio = filtered_portfolio.sort_values('ROE', ascending=False)
        
        # Add applied filters information
        if filters_applied:
            st.info(f"Applied filters: {', '.join(filters_applied)}")
        
        return df_portfolio
    
    def calculate_correlation_significance(self, df):
        """
        Calculate correlation and statistical significance between financial ratios.
        This is a backend calculation that returns actionable insights rather than raw statistics.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing financial ratios
            
        Returns:
        --------
        dict
            Dictionary containing correlation insights and significant relationships
        """
        # Financial ratios to analyze
        financial_ratios = ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE', 
                        'Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets']
        
        # Filter columns that exist in the DataFrame
        available_ratios = [col for col in financial_ratios if col in df.columns]
        
        # Calculate correlation matrix (backend calculation)
        correlation_matrix = df[available_ratios].corr()
        
        # Calculate statistical significance (p-values) - backend calculation
        p_values = pd.DataFrame(index=available_ratios, columns=available_ratios)
        
        for i in available_ratios:
            for j in available_ratios:
                if i != j:
                    # Remove missing values
                    valid_data = df[[i, j]].dropna()
                    
                    if len(valid_data) > 2:  # Need at least 3 points for regression
                        # Add constant for regression
                        X = sm.add_constant(valid_data[i])
                        
                        # Fit regression model
                        model = sm.OLS(valid_data[j], X).fit()
                        
                        # Get p-value for the coefficient
                        p_values.loc[i, j] = model.pvalues[1]
                    else:
                        p_values.loc[i, j] = np.nan
                else:
                    p_values.loc[i, j] = 0.0  # p-value for self-correlation is 0
        
        # Extract actionable insights from the statistical analysis
        insights = {
            'strong_positive_correlations': [],
            'strong_negative_correlations': [],
            'significant_relationships': [],
            'key_drivers': {},
            'correlation_matrix': correlation_matrix,  # Keep for backend reference
            'p_values': p_values  # Keep for backend reference
        }
        
        # Find strong and statistically significant correlations
        for i in available_ratios:
            key_correlations = []
            
            for j in available_ratios:
                if i != j:
                    corr_value = correlation_matrix.loc[i, j]
                    p_value = p_values.loc[i, j]
                    
                    # Check if correlation is strong and significant
                    if not pd.isna(corr_value) and not pd.isna(p_value):
                        if abs(corr_value) >= 0.7 and p_value < 0.05:
                            relationship = {
                                'ratio_1': i,
                                'ratio_2': j,
                                'correlation': corr_value,
                                'p_value': p_value,
                                'strength': 'Strong' if abs(corr_value) >= 0.7 else 'Moderate'
                            }
                            
                            if corr_value > 0:
                                insights['strong_positive_correlations'].append(relationship)
                            else:
                                insights['strong_negative_correlations'].append(relationship)
                            
                            insights['significant_relationships'].append(relationship)
                            key_correlations.append((j, abs(corr_value)))
            
            # Identify key drivers for each ratio
            if key_correlations:
                # Sort by correlation strength
                key_correlations.sort(key=lambda x: x[1], reverse=True)
                insights['key_drivers'][i] = [ratio for ratio, _ in key_correlations[:2]]
        
        return insights

    def predict_investment_recommendation(self, company_data):
        """
        Predict investment recommendation (Buy, Hold, Sell) for a company.
        
        Parameters:
        -----------
        company_data : dict
            Dictionary containing company financial data
            
        Returns:
        --------
        tuple
            Tuple containing recommendation and confidence score
        """
        # Check if model is available
        if 'recommendation' not in self.models or self.models['recommendation'] is None:
            # Fallback to rule-based recommendation
            return self._rule_based_recommendation(company_data)
        
        try:
            # Extract features for prediction
            features = self._extract_features(company_data)
            
            # Make prediction
            prediction = self.models['recommendation'].predict([features])[0]
            probabilities = self.models['recommendation'].predict_proba([features])[0]
            
            # Get confidence score
            confidence = max(probabilities) * 100
            
            # Map prediction to recommendation
            recommendation_map = {0: 'Sell', 1: 'Hold', 2: 'Buy'}
            recommendation = recommendation_map.get(prediction, 'Hold')
            
            return recommendation, confidence
        
        except Exception as e:
            st.warning(f"Error predicting recommendation: {e}")
            # Fallback to rule-based recommendation
            return self._rule_based_recommendation(company_data)
    
    def _rule_based_recommendation(self, company_data):
        """
        Generate a rule-based investment recommendation.
        
        Parameters:
        -----------
        company_data : dict
            Dictionary containing company financial data
            
        Returns:
        --------
        tuple
            Tuple containing recommendation and confidence score
        """
        # Extract key metrics
        roe = company_data.get('ROE', 0)
        npm = company_data.get('Net Profit Margin', 0)
        debt_equity = company_data.get('Debt-to-Equity', 0)
        
        # Calculate score
        score = 0
        
        # ROE score (0-40 points)
        if roe >= 0.15:  # 15% or higher
            score += 40
        elif roe >= 0.10:  # 10-15%
            score += 30
        elif roe >= 0.05:  # 5-10%
            score += 20
        elif roe > 0:  # 0-5%
            score += 10
        
        # Net Profit Margin score (0-30 points)
        if npm >= 0.15:  # 15% or higher
            score += 30
        elif npm >= 0.10:  # 10-15%
            score += 25
        elif npm >= 0.05:  # 5-10%
            score += 15
        elif npm > 0:  # 0-5%
            score += 5
        
        # Debt-to-Equity score (0-30 points)
        if debt_equity <= 0.5:  # Very low debt
            score += 30
        elif debt_equity <= 1.0:  # Low debt
            score += 25
        elif debt_equity <= 1.5:  # Moderate debt
            score += 15
        elif debt_equity <= 2.0:  # High debt
            score += 5
        
        # Determine recommendation
        if score >= 70:
            return 'Buy', score
        elif score >= 40:
            return 'Hold', score
        else:
            return 'Sell', score
    
    def _extract_features(self, company_data):
        """
        Extract features for model prediction.
        
        Parameters:
        -----------
        company_data : dict
            Dictionary containing company financial data
            
        Returns:
        --------
        list
            List of features for model prediction
        """
        # Extract numerical features
        features = [
            company_data.get('ROE', 0),
            company_data.get('Net Profit Margin', 0),
            company_data.get('ROA', 0),
            company_data.get('Gross Margin', 0),
            company_data.get('Current Ratio', 0),
            company_data.get('Debt-to-Equity', 0),
            company_data.get('Debt-to-Assets', 0)
        ]
        
        # Add company encoding if available
        if 'company' in self.encoders and self.encoders['company'] is not None:
            company = company_data.get('Company', '')
            try:
                company_encoding = self.encoders['company'].transform([company])[0]
                features.append(company_encoding)
            except:
                features.append(0)
        
        return features
    
    def predict_future_roe(self, company_data):
        """
        Predict future ROE for a company.
        
        Parameters:
        -----------
        company_data : dict
            Dictionary containing company financial data
            
        Returns:
        --------
        float
            Predicted ROE value
        """
        # Check if model is available
        if 'roe_prediction' not in self.models or self.models['roe_prediction'] is None:
            # Fallback to simple prediction
            return company_data.get('ROE', 0) * 1.05  # Assume 5% growth
        
        try:
            # Extract features for prediction
            features = self._extract_features(company_data)
            
            # Make prediction
            prediction = self.models['roe_prediction'].predict([features])[0]
            
            return prediction
        
        except Exception as e:
            st.warning(f"Error predicting ROE: {e}")
            # Fallback to simple prediction
            return company_data.get('ROE', 0) * 1.05  # Assume 5% growth

# Main application
def main():
    # Load data
    df = load_financial_data()
    
    # Load AI models
    models, encoders, model_status = load_ai_models()
    
    # Initialize Financial AI
    financial_ai = FinancialAI(models, encoders)
    
    # Sidebar
    st.sidebar.title("Financial Analysis Options")
    
    # Add navigation menu
    st.sidebar.markdown("---")
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("", [
        "Company Analysis", 
        "Portfolio Builder", 
        "Company Comparison", 
        "Industry Analysis"
    ])
    
    # Display content based on selected page
    if page == "Company Analysis":
        # Company selection
        companies = sorted(df['Company'].unique())
        selected_company = st.sidebar.selectbox("Select Company", companies)
        
        # Period type selection
        period_types = sorted(df['Period_Type'].unique())
        selected_period_type = st.sidebar.selectbox("Select Period Type", period_types)
        
        # Filter data
        company_data = df[(df['Company'] == selected_company) & (df['Period_Type'] == selected_period_type)]
        
        # Sort by date
        if 'Date' in company_data.columns:
            company_data = company_data.sort_values('Date')
        
        # Period selection
        if not company_data.empty:
            periods = company_data['Period'].unique()
            selected_period = st.sidebar.selectbox("Select Period", periods)
            
            # Get data for selected period
            period_data = company_data[company_data['Period'] == selected_period].iloc[0].to_dict()
        else:
            st.error(f"No data available for {selected_company} with period type {selected_period_type}")
            return
            
        # Display company overview
        st.header(f"{selected_company} Financial Analysis")
        st.subheader(f"Period: {selected_period}")
        
        # Financial metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("ROE", f"{period_data.get('ROE', 0)*100:.2f}%")
        
        with col2:
            st.metric("Net Profit Margin", f"{period_data.get('Net Profit Margin', 0)*100:.2f}%")
        
        with col3:
            st.metric("Debt-to-Equity", f"{period_data.get('Debt-to-Equity', 0):.2f}")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["ROE Trend", "Profitability", "Liquidity & Leverage", "Statistical Analysis"])
        
        with tab1:
            # ROE Trend
            st.subheader(f"{selected_company} ROE Trend")
            
            # Get historical data
            historical_data = company_data.sort_values('Date')
            
            # Create ROE trend chart
            fig = px.line(historical_data, x='Period', y='ROE', 
                         title=f"{selected_company} ROE Trend")
            fig.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Profitability
            st.subheader(f"{selected_company} Profitability")
            
            # Create profitability chart
            fig = px.line(historical_data, x='Period', y=['Gross Margin', 'Net Profit Margin'], 
                         title=f"{selected_company} Profitability Metrics")
            fig.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Liquidity & Leverage
            st.subheader(f"{selected_company} Liquidity & Leverage")
            
            # Create liquidity & leverage chart
            fig = px.line(historical_data, x='Period', y=['Current Ratio', 'Debt-to-Equity'], 
                         title=f"{selected_company} Liquidity & Leverage")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Statistical Analysis - Now showing actionable insights instead of raw statistics
            st.subheader("Financial Relationship Insights")
            
            # Calculate correlation insights using the enhanced backend function
            insights = financial_ai.calculate_correlation_significance(df)
            
            # Display key insights instead of raw correlation matrix
            st.write("### Key Financial Relationships")
            
            # Show key drivers for important metrics
            if insights['key_drivers']:
                st.write("#### Key Drivers of Financial Performance")
                for ratio, drivers in insights['key_drivers'].items():
                    if drivers:
                        st.write(f"**{ratio}** is primarily influenced by: {', '.join(drivers)}")
            
            # Show strong positive correlations
            if insights['strong_positive_correlations']:
                st.write("#### Strong Positive Relationships")
                for rel in insights['strong_positive_correlations']:
                    st.write(f"• **{rel['ratio_1']}** and **{rel['ratio_2']}** move together " +
                             f"(correlation: {rel['correlation']:.2f})")
            
            # Show strong negative correlations
            if insights['strong_negative_correlations']:
                st.write("#### Strong Negative Relationships")
                for rel in insights['strong_negative_correlations']:
                    st.write(f"• **{rel['ratio_1']}** and **{rel['ratio_2']}** move in opposite directions " +
                             f"(correlation: {rel['correlation']:.2f})")
            
            # Show correlation heatmap (simplified visualization)
            if st.checkbox("Show Correlation Heatmap"):
                fig = px.imshow(insights['correlation_matrix'], 
                               text_auto='.2f', 
                               color_continuous_scale='RdBu_r',
                               title="Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
            
            # Show significant relationships in a more user-friendly format
            if insights['significant_relationships']:
                if st.checkbox("Show Detailed Statistical Relationships"):
                    st.subheader("Statistically Significant Relationships")
                    sig_df = pd.DataFrame([{
                        'Ratio 1': rel['ratio_1'],
                        'Ratio 2': rel['ratio_2'],
                        'Relationship': f"{rel['strength']} {'Positive' if rel['correlation'] > 0 else 'Negative'}",
                        'Correlation': f"{rel['correlation']:.2f}"
                    } for rel in insights['significant_relationships']])
                    st.dataframe(sig_df)
            else:
                st.write("No statistically significant relationships found in the data.")
        
        # AI-Powered Analysis
        st.header("AI-Powered Analysis")
        
        # Investment recommendation
        recommendation, confidence = financial_ai.predict_investment_recommendation(period_data)
        
        # Display recommendation
        col1, col2 = st.columns(2)
        
        with col1:
            recommendation_class = f"recommendation-{recommendation.lower()}"
            st.markdown(f"""
            <div class="{recommendation_class}">
                <h3>Investment Recommendation: {recommendation}</h3>
                <p>Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Company status
            roe = period_data.get('ROE', 0)
            npm = period_data.get('Net Profit Margin', 0)
            debt_equity = period_data.get('Debt-to-Equity', 0)
            
            # Determine status
            if roe >= 0.15 and npm >= 0.10 and debt_equity <= 1.5:
                status = "Excellent"
            elif roe >= 0.10 and npm >= 0.05 and debt_equity <= 2.0:
                status = "Good"
            elif roe >= 0.05 and npm > 0:
                status = "Fair"
            else:
                status = "Poor"
            
            # Predict future ROE
            predicted_roe = financial_ai.predict_future_roe(period_data)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Company Status: {status}</h3>
                <p>Predicted ROE: {predicted_roe*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Historical Performance
        st.header("Historical Performance")
        
        # Create tabs for different historical analyses
        tab1, tab2, tab3, tab4 = st.tabs(["ROE Trend", "Profitability", "Liquidity & Leverage", "Statistical Analysis"])
        
        with tab1:
            # ROE Trend
            st.subheader(f"{selected_company} ROE Trend")
            
            # Create ROE trend chart
            fig = px.line(historical_data, x='Date', y='ROE', 
                         title=f"{selected_company} ROE Trend")
            fig.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Portfolio Builder":
        # Portfolio Builder
        st.header("Portfolio Builder")
        
        # Set investment criteria
        st.subheader("Set Investment Criteria")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_roi = st.slider("Minimum ROI (%)", 0.0, 20.0, 8.0) / 100
        
        with col2:
            max_risk = st.slider("Maximum Risk (Debt-to-Equity)", 0.0, 3.0, 1.5)
        
        with col3:
            min_growth = st.slider("Minimum Growth (Net Profit Margin %)", 0.0, 15.0, 5.0) / 100
        
        # Build portfolio
        criteria = {
            'min_roi': min_roi,
            'max_risk': max_risk,
            'min_growth': min_growth
        }
        
        portfolio = financial_ai.build_portfolio(df, criteria)
        
        # Display portfolio
        st.subheader("Recommended Portfolio")
        
        if not portfolio.empty:
            # Create a clean display dataframe
            display_df = portfolio[['Company', 'Period', 'ROE', 'Net Profit Margin', 'Debt-to-Equity']].copy()
            
            # Format percentage columns
            display_df['ROE'] = display_df['ROE'].apply(lambda x: f"{x*100:.2f}%")
            display_df['Net Profit Margin'] = display_df['Net Profit Margin'].apply(lambda x: f"{x*100:.2f}%")
            
            # Rename columns for display
            display_df.columns = ['Company', 'Period', 'ROE', 'Net Profit Margin', 'Debt-to-Equity']
            
            # Display table
            st.dataframe(display_df)
            
            # Portfolio visualization
            st.subheader("Portfolio Visualization")
            
            # ROE comparison
            fig = px.bar(
                portfolio,
                x='Company',
                y='ROE',
                title="Return on Equity (ROE) by Company",
                color='Company'
            )
            fig.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk-Return scatter plot
            fig = px.scatter(
                portfolio,
                x='Debt-to-Equity',
                y='ROE',
                size='Net Profit Margin',
                color='Company',
                title="Risk vs. Return",
                labels={
                    'Debt-to-Equity': 'Risk (Debt-to-Equity)',
                    'ROE': 'Return (ROE)',
                    'Net Profit Margin': 'Profit Margin'
                }
            )
            fig.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No companies match the selected criteria. Try adjusting your parameters.")
        
        # Add new company
        st.header("Add New Company")
        
        # File uploader
        st.write("Upload CSV file with new company data")
        uploaded_file = st.file_uploader("", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Load data
                new_company_data = pd.read_csv(uploaded_file)
                
                # Check if Company column exists
                if 'Company' not in new_company_data.columns:
                    company_name = st.text_input("Enter company name")
                    if company_name:
                        # Add company data
                        data_dir = os.path.join(os.getcwd(), 'data')
                        loader = DataLoader(data_dir)
                        
                        # Load existing data
                        if os.path.exists(os.path.join(data_dir, 'processed', 'processed_financial_ratios.csv')):
                            loader.processed_data = pd.read_csv(os.path.join(data_dir, 'processed', 'processed_financial_ratios.csv'))
                        
                        # Add new company
                        success = loader.add_company_data(new_company_data, company_name)
                        
                        if success:
                            st.success(f"Company {company_name} added successfully")
                            
                            # Update session state
                            st.session_state.financial_data = loader.processed_data
                            
                            # Reload page
                            st.experimental_rerun()
                else:
                    # Add company data
                    data_dir = os.path.join(os.getcwd(), 'data')
                    loader = DataLoader(data_dir)
                    
                    # Load existing data
                    if os.path.exists(os.path.join(data_dir, 'processed', 'processed_financial_ratios.csv')):
                        loader.processed_data = pd.read_csv(os.path.join(data_dir, 'processed', 'processed_financial_ratios.csv'))
                    
                    # Add new company
                    success = loader.add_company_data(new_company_data)
                    
                    if success:
                        st.success(f"Company added successfully")
                        
                        # Update session state
                        st.session_state.financial_data = loader.processed_data
                        
                        # Reload page
                        st.experimental_rerun()
            
            except Exception as e:
                st.error(f"Error adding company: {e}")
    
    elif page == "Company Comparison":
        # Company Comparison
        company_comparison_component(df)
    
    elif page == "Industry Analysis":
        # Industry Analysis
        industry_analysis_component(df)

# Run the app
if __name__ == "__main__":
    main()
