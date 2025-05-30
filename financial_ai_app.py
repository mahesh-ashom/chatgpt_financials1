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

# DataLoader class for robust data handling
class DataLoader:
    """
    A flexible data loader for financial ratio data that:
    1. Loads data from CSV files
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
        """
        if self.raw_data is None or self.raw_data.empty:
            st.warning("No raw data available for processing")
            return
        
        try:
            st.write("Processing data...")
            processed_data = self.raw_data.copy()
            
            # Clean column names - strip whitespace
            processed_data.columns = [col.strip() if isinstance(col, str) else col for col in processed_data.columns]
            
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
                            if pd.notna(row['Year']) and pd.notna(row['Quarter']) else None, axis=1)
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
            if self.processed_data is not None and not self.processed_data.empty:
                if company_name in self.processed_data['Company'].unique():
                    st.warning(f"Company '{company_name}' already exists. Use update_company_data to update.")
                    return False
            
            # Ensure required columns exist
            required_columns = ['Period', 'Period_Type']
            missing_columns = [col for col in required_columns if col not in company_data.columns]
            
            if missing_columns:
                st.warning(f"Missing required columns: {', '.join(missing_columns)}")
                
                # Try to infer missing columns
                if 'Period' not in company_data.columns:
                    if 'Year' in company_data.columns and 'Quarter' in company_data.columns:
                        company_data['Period'] = company_data.apply(
                            lambda row: f"{int(float(row['Year']))}-Q{int(float(row['Quarter']))}" 
                            if pd.notna(row['Year']) and pd.notna(row['Quarter']) else None, axis=1)
                    elif 'Year' in company_data.columns:
                        company_data['Period'] = company_data['Year'].astype(str)
                
                if 'Period_Type' not in company_data.columns:
                    if 'Quarter' in company_data.columns:
                        # Mark records with Quarter=0 as Annual, others as Quarterly
                        company_data['Period_Type'] = company_data['Quarter'].apply(
                            lambda q: 'Annual' if pd.notna(q) and float(q) == 0 else 'Quarterly'
                        )
                    else:
                        company_data['Period_Type'] = 'Annual'
            
            # Convert date columns if they exist
            date_columns = ['Date', 'date']
            for col in date_columns:
                if col in company_data.columns:
                    company_data[col] = pd.to_datetime(company_data[col], errors='coerce')
                    
                    # Rename to standardized column name
                    if col != 'Date':
                        company_data['Date'] = company_data[col]
                        company_data = company_data.drop(columns=[col])
            
            # If Date column doesn't exist, try to create it
            if 'Date' not in company_data.columns:
                if 'Year' in company_data.columns and 'Quarter' in company_data.columns:
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
                    
                    company_data['Date'] = company_data.apply(quarter_to_date, axis=1)
                
                elif 'Year' in company_data.columns:
                    # Create dates for years (January 1st) with error handling
                    company_data['Date'] = pd.to_datetime(company_data['Year'].astype(str) + '-01-01', errors='coerce')
            
            # Clean financial ratio columns
            financial_columns = ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE', 
                               'Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets']
            
            for col in financial_columns:
                if col in company_data.columns:
                    # If column is string type, clean it
                    if company_data[col].dtype == 'object':
                        # Remove % signs, commas, etc.
                        company_data[col] = company_data[col].astype(str)
                        company_data[col] = company_data[col].str.replace('%', '', regex=False)
                        company_data[col] = company_data[col].str.replace(',', '', regex=False)
                        company_data[col] = company_data[col].str.strip()
                        
                    # Convert to numeric
                    company_data[col] = pd.to_numeric(company_data[col], errors='coerce')
                    
                    # Check if values are percentages (>1) and need conversion to decimal
                    if col in ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE'] and company_data[col].max() > 1:
                        # Check if values look like percentages (e.g., 15.7 for 15.7%)
                        if company_data[col].max() <= 100:
                            company_data[col] = company_data[col] / 100
                            st.write(f"Converted {col} from percentage to decimal format")
            
            # Combine with existing data
            if self.processed_data is None or self.processed_data.empty:
                self.processed_data = company_data
            else:
                # Ensure columns match
                existing_columns = set(self.processed_data.columns)
                new_columns = set(company_data.columns)
                
                # Add missing columns to company_data
                for col in existing_columns - new_columns:
                    company_data[col] = None
                
                # Add missing columns to processed_data
                for col in new_columns - existing_columns:
                    self.processed_data[col] = None
                
                # Combine data
                self.processed_data = pd.concat([self.processed_data, company_data], ignore_index=True)
            
            # Update companies list
            self.companies = sorted(self.processed_data['Company'].unique().tolist())
            
            # Update available ratios
            metadata_columns = ['Period', 'Period_Type', 'Year', 'Quarter', 'Company', 'Date']
            self.available_ratios = [col for col in self.processed_data.columns 
                                   if col not in metadata_columns]
            
            # Save processed data if directory exists
            if self.data_dir:
                self._save_processed_data()
            
            st.success(f"Company '{company_name}' added successfully")
            return True
        
        except Exception as e:
            st.error(f"Error adding company data: {e}")
            return False
    
    def get_company_data(self, company_name=None, period_type=None):
        """
        Get data for a specific company and period type.
        
        Parameters:
        -----------
        company_name : str, optional
            Name of the company to get data for
        period_type : str, optional
            Type of period to get data for ('Annual' or 'Quarterly')
            
        Returns:
        --------
        pandas.DataFrame
            Data for the specified company and period type
        """
        if self.processed_data is None or self.processed_data.empty:
            st.warning("No processed data available")
            return None
        
        try:
            # Filter by company if specified
            if company_name:
                if company_name not in self.companies:
                    st.warning(f"Company '{company_name}' not found")
                    return None
                
                data = self.processed_data[self.processed_data['Company'] == company_name]
            else:
                data = self.processed_data
            
            # Filter by period type if specified
            if period_type:
                if period_type not in ['Annual', 'Quarterly']:
                    st.warning(f"Invalid period type: {period_type}")
                    return None
                
                data = data[data['Period_Type'] == period_type]
            
            return data
        
        except Exception as e:
            st.error(f"Error getting company data: {e}")
            return None

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
            'data.csv',
            'SavolaAlmaraiNADECFinancialRatiosCSV.csv.csv'
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
        
        # Initialize DataLoader
        loader = DataLoader()
        
        # Load and process the data
        loader.load_csv(data=df)
        
        # Return processed data
        return loader.processed_data
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration purposes"""
    companies = ['Almarai', 'Savola', 'NADEC']
    years = [2020, 2021, 2022, 2023]
    quarters = [1, 2, 3, 4]
    
    data = []
    
    # Base financial ratios for each company (realistic Saudi market values)
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
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Initialize DataLoader
    loader = DataLoader()
    
    # Load and process the data
    loader.load_csv(data=df)
    
    # Return processed data
    return loader.processed_data

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
            return max(0, min(prediction, 1))  # Bound between 0 and 100%
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
        """Fallback ROE prediction using financial relationships"""
        roa = company_data.get('ROA', 0)
        npm = company_data.get('Net Profit Margin', 0)
        equity_multiplier = 1 + company_data.get('Debt-to-Equity', 0)
        
        # ROE = ROA × Equity Multiplier (simplified DuPont formula)
        predicted_roe = roa * equity_multiplier
        
        # Alternative calculation using profit margin relationship
        if predicted_roe == 0:
            predicted_roe = npm * 1.2  # Rough approximation
        
        return max(0, min(predicted_roe, 1))
    
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
        """Fallback status prediction"""
        roe = company_data.get('ROE', 0)
        npm = company_data.get('Net Profit Margin', 0)
        
        if roe > 0.15 and npm > 0.15:
            return "Excellent"
        elif roe > 0.10 and npm > 0.10:
            return "Good"
        elif roe > 0.05 and npm > 0.05:
            return "Average"
        else:
            return "Poor"
    
    def _calculate_investment_score(self, data):
        """Calculate investment score based on financial metrics"""
        score = 0
        
        # ROE scoring (35% weight)
        roe = data.get('ROE', 0)
        if roe > 0.20: score += 35
        elif roe > 0.15: score += 30
        elif roe > 0.12: score += 25
        elif roe > 0.08: score += 15
        elif roe > 0.05: score += 8
        elif roe > 0.02: score += 3
        
        # ROA scoring (25% weight)
        roa = data.get('ROA', 0)
        if roa > 0.12: score += 25
        elif roa > 0.08: score += 20
        elif roa > 0.06: score += 15
        elif roa > 0.04: score += 10
        elif roa > 0.02: score += 5
        
        # Net Profit Margin scoring (20% weight)
        npm = data.get('Net Profit Margin', 0)
        if npm > 0.15: score += 20
        elif npm > 0.10: score += 15
        elif npm > 0.05: score += 10
        elif npm > 0.02: score += 5
        
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
            'confidence': confidence,
            'company_status': company_status
        }

    def build_portfolio(self, df, criteria):
        """
        Build investment portfolio based on specified criteria.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing financial ratios and investment metrics
        criteria : dict
            Dictionary containing portfolio criteria
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing selected portfolio companies
        """
        # Create a copy of the DataFrame
        df_portfolio = df.copy()
        
        # Filter by period type (use most recent data)
        if 'Period_Type' in df_portfolio.columns:
            # Get the most recent date for each company
            most_recent = df_portfolio.groupby('Company')['Date'].max().reset_index()
            
            # Merge to get only the most recent data for each company
            df_portfolio = df_portfolio.merge(most_recent, on=['Company', 'Date'])
        
        # Apply criteria filters
        if 'min_roi' in criteria:
            df_portfolio = df_portfolio[df_portfolio['ROE'] >= criteria['min_roi']]
        
        if 'max_risk' in criteria:
            # Use Debt-to-Equity as a risk measure
            df_portfolio = df_portfolio[df_portfolio['Debt-to-Equity'] <= criteria['max_risk']]
        
        if 'min_growth' in criteria:
            # For growth, we would ideally need historical data to calculate growth rate
            # As a proxy, we can use Net Profit Margin
            df_portfolio = df_portfolio[df_portfolio['Net Profit Margin'] >= criteria['min_growth']]
        
        if 'industry' in criteria:
            # This would require industry classification data
            # For now, we'll just filter by company if it matches the industry name
            df_portfolio = df_portfolio[df_portfolio['Company'].str.contains(criteria['industry'], case=False)]
        
        # Sort by ROE (descending) as default ranking
        df_portfolio = df_portfolio.sort_values('ROE', ascending=False)
        
        return df_portfolio

    def calculate_correlation_significance(self, df):
        """
        Calculate correlation and statistical significance between financial ratios.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing financial ratios
            
        Returns:
        --------
        tuple
            Tuple containing correlation matrix and statistical significance DataFrame
        """
        # Financial ratios to analyze
        financial_ratios = ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE', 
                        'Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets']
        
        # Filter columns that exist in the DataFrame
        available_ratios = [col for col in financial_ratios if col in df.columns]
        
        # Calculate correlation matrix
        correlation_matrix = df[available_ratios].corr()
        
        # Calculate statistical significance (p-values)
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
        
        return correlation_matrix, p_values

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
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("ROE", f"{period_data.get('ROE', 0)*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Net Profit Margin", f"{period_data.get('Net Profit Margin', 0)*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Debt-to-Equity", f"{period_data.get('Debt-to-Equity', 0):.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # AI Analysis
    st.header("AI-Powered Analysis")
    
    # Get AI predictions
    analysis = financial_ai.comprehensive_analysis(period_data)
    
    # Display predictions
    col1, col2 = st.columns(2)
    
    with col1:
        recommendation = analysis['investment_recommendation']
        confidence = analysis['confidence']
        
        if recommendation in ['Strong Buy', 'Buy']:
            st.markdown(f'<div class="recommendation-buy">', unsafe_allow_html=True)
        elif recommendation == 'Hold':
            st.markdown(f'<div class="recommendation-hold">', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="recommendation-sell">', unsafe_allow_html=True)
        
        st.subheader(f"Investment Recommendation: {recommendation}")
        st.write(f"Confidence: {confidence*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader(f"Company Status: {analysis['company_status']}")
        st.write(f"Predicted ROE: {analysis['predicted_roe']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Historical performance
    st.header("Historical Performance")
    
    # Filter data for the selected company
    historical_data = df[df['Company'] == selected_company].sort_values('Date')
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["ROE Trend", "Profitability", "Liquidity & Leverage", "Statistical Analysis"])
    
    with tab1:
        # ROE Trend
        fig = px.line(historical_data, x='Period', y='ROE', title=f"{selected_company} ROE Trend")
        fig.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Profitability metrics
        fig = px.line(historical_data, x='Period', y=['Gross Margin', 'Net Profit Margin'], 
                     title=f"{selected_company} Profitability Metrics")
        fig.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Liquidity and leverage
        fig = px.line(historical_data, x='Period', y=['Current Ratio', 'Debt-to-Equity'], 
                     title=f"{selected_company} Liquidity & Leverage")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Statistical Analysis
        st.subheader("Correlation & Statistical Significance")
        
        # Calculate correlation and significance
        correlation_matrix, p_values = financial_ai.calculate_correlation_significance(df)
        
        # Display correlation heatmap
        fig = px.imshow(correlation_matrix, 
                       text_auto='.2f', 
                       color_continuous_scale='RdBu_r',
                       title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # Display significant relationships
        st.subheader("Statistically Significant Relationships (p < 0.05)")
        significant_relations = []
        
        for i in p_values.index:
            for j in p_values.columns:
                if i != j and not pd.isna(p_values.loc[i, j]) and p_values.loc[i, j] < 0.05:
                    corr = correlation_matrix.loc[i, j]
                    significant_relations.append({
                        'Variable 1': i,
                        'Variable 2': j,
                        'Correlation': corr,
                        'p-value': p_values.loc[i, j]
                    })
        
        if significant_relations:
            sig_df = pd.DataFrame(significant_relations)
            st.dataframe(sig_df)
        else:
            st.write("No statistically significant relationships found.")
    
    # Portfolio Builder
    st.header("Portfolio Builder")
    
    # Portfolio criteria inputs
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
    
    if len(portfolio) > 0:
        # Format percentages for display
        display_portfolio = portfolio.copy()
        for col in ['ROE', 'Net Profit Margin', 'ROA', 'Gross Margin']:
            if col in display_portfolio.columns:
                display_portfolio[col] = display_portfolio[col].apply(lambda x: f"{x*100:.2f}%")
        
        # Select columns to display
        display_cols = ['Company', 'Period', 'ROE', 'Net Profit Margin', 'Debt-to-Equity', 'Current Ratio']
        display_cols = [col for col in display_cols if col in display_portfolio.columns]
        
        st.dataframe(display_portfolio[display_cols])
        
        # Portfolio composition chart
        fig = px.pie(portfolio, names='Company', values='ROE', title="Portfolio Composition")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No companies match the selected criteria. Try adjusting your parameters.")
    
    # Add new company section
    st.header("Add New Company")
    
    # File uploader for new company data
    uploaded_file = st.file_uploader("Upload CSV file with new company data", type="csv")
    
    if uploaded_file is not None:
        # Read the uploaded file
        new_company_data = pd.read_csv(uploaded_file)
        
        # Display the uploaded data
        st.subheader("Uploaded Data Preview")
        st.dataframe(new_company_data.head())
        
        # Company name input
        new_company_name = st.text_input("Company Name (leave blank if included in CSV)")
        
        # Add button
        if st.button("Add Company"):
            # Initialize DataLoader
            loader = DataLoader()
            
            # Load existing data
            loader.load_csv(data=df)
            
            # Add new company
            if new_company_name:
                success = loader.add_company_data(new_company_data, new_company_name)
            else:
                success = loader.add_company_data(new_company_data)
            
            if success:
                st.success("Company added successfully! Please refresh the page to see the updated data.")
            else:
                st.error("Failed to add company. Please check the data format.")

# Run the app
if __name__ == "__main__":
    main()
