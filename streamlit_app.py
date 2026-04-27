import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import os

st.set_page_config(
    page_title="Rossman Sales Predictor",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    h2 {
        color: #333;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and artifacts
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('rossman_model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('encoder.pkl')
        store_df = pd.read_csv('C:/Users/nanda/Downloads/store.csv')
        return model, scaler, encoder, store_df
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {str(e)}")
        st.info("Please run the save artifacts code in your Jupyter notebook first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading artifacts: {str(e)}")
        st.stop()

# Feature engineering functions
def split_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['WeekOfYear'] = df.Date.dt.isocalendar().week

def comp_months(df):
    df['CompetitionOpen'] = (
        (df['Year'] - df['CompetitionOpenSinceYear']) * 12 +
        (df['Month'] - df['CompetitionOpenSinceMonth'])
    )
    df['CompetitionOpen'] = df['CompetitionOpen'].apply(lambda x: 0 if pd.isna(x) or x < 0 else x)

def promo_cols(df):
    df['Promo2Open'] = (
        (df['Year'] - df['Promo2SinceYear']) * 12 +
        (df['WeekOfYear'] - df['Promo2SinceWeek']) // 4
    )
    df['Promo2Open'] = df['Promo2Open'].apply(lambda x: 0 if pd.isna(x) or x < 0 else x)
    df['Promo2Open'] = df['Promo2Open'] * df['Promo2']
    
    month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    
    def is_promo_month(row):
        months = str(row['PromoInterval']).split(',') if pd.notna(row['PromoInterval']) else []
        return int(row['Promo2'] and month_map[row['Month']] in months)
    
    df['IsPromo2Month'] = df.apply(is_promo_month, axis=1)

def make_prediction(sample_input, model, scaler, encoder, store_df):
    """Process input and make prediction"""
    numeric_cols = ['Store', 'Promo', 'SchoolHoliday',
                    'CompetitionDistance', 'CompetitionOpen', 'Promo2', 'Promo2Open',
                    'Day', 'Month', 'Year', 'WeekOfYear', 'IsPromo2Month']
    
    categorical_cols = ['DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment']
    
    # Create DataFrame
    input_df = pd.DataFrame([sample_input])
    merged_input_df = input_df.merge(store_df, on='Store')
    
    # Feature engineering
    split_date(merged_input_df)
    comp_months(merged_input_df)
    promo_cols(merged_input_df)
    
    # Handle missing values
    max_distance = 75860  # From training data
    if merged_input_df['CompetitionDistance'].isna().any():
        merged_input_df['CompetitionDistance'].fillna(max_distance, inplace=True)
    
    # Scale numeric columns
    merged_input_df[numeric_cols] = scaler.transform(merged_input_df[numeric_cols])
    
    # Encode categorical columns
    merged_input_df[categorical_cols] = merged_input_df[categorical_cols].astype(str)
    encoded_cols_list = list(encoder.get_feature_names_out(categorical_cols))
    merged_input_df[encoded_cols_list] = encoder.transform(merged_input_df[categorical_cols])
    merged_input_df = merged_input_df.drop(columns=categorical_cols)
    
    # Build input and predict
    X_input = merged_input_df[model.feature_names_in_]
    prediction = model.predict(X_input)
    
    return max(0, prediction[0])

# Load artifacts
model, scaler, encoder, store_df = load_artifacts()

# Header
st.markdown("<h1>🏪 Rossman Sales Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 1.1rem;'>Predict daily sales for Rossman stores</p>", 
            unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "ℹ️ Information", "📈 Analytics"])

with tab1:
    st.markdown("<h2>Prediction Input</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Store Information")
        store_id = st.number_input(
            "Store ID",
            min_value=1,
            max_value=1115,
            value=5,
            help="Select a store number from 1 to 1115"
        )
        
        date = st.date_input(
            "Date",
            value=datetime(2015, 9, 17),
            help="Select the date for prediction"
        )
        
        # Automatically calculate day of week from the date (1=Monday, 7=Sunday)
        day_of_week = (datetime.combine(date, datetime.min.time()).weekday() % 7) + 1
    
    with col2:
        st.markdown("### Promotions & Holidays")
        promo = st.checkbox("Promotion Active", value=True)
        school_holiday = st.checkbox("School Holiday", value=False)
        
        state_holiday = st.selectbox(
            "State Holiday",
            options=['0', 'a', 'b', 'c'],
            format_func=lambda x: {'0': 'None', 'a': 'Holiday A', 'b': 'Holiday B', 'c': 'Holiday C'}[x],
            help="Select state holiday type"
        )
    
    # Create sample input
    sample_input = {
        'Store': store_id,
        'DayOfWeek': day_of_week,
        'Promo': int(promo),
        'Date': str(date),
        'Open': 1,
        'StateHoliday': state_holiday,
        'SchoolHoliday': int(school_holiday)
    }
    
    # Predict button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🎯 Predict Sales", use_container_width=True, type="primary")
    
    if predict_button:
        try:
            with st.spinner("🔄 Making prediction..."):
                prediction = make_prediction(sample_input, model, scaler, encoder, store_df)
            
            # Display prediction
            st.markdown(f"""
                <div class='prediction-box'>
                    <h3>Predicted Sales</h3>
                    <div class='prediction-value'>€ {prediction:,.2f}</div>
                    <p style='font-size: 0.9rem; opacity: 0.9;'>for Store {store_id} on {date.strftime('%B %d, %Y')}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display input summary
            st.markdown("<h3>📋 Input Summary</h3>", unsafe_allow_html=True)
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.metric("Store ID", store_id)
                st.metric("Date", date.strftime("%B %d, %Y"))
                st.metric("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day_of_week-1])
            
            with summary_col2:
                st.metric("Promotion Active", "✅ Yes" if promo else "❌ No")
                st.metric("School Holiday", "✅ Yes" if school_holiday else "❌ No")
                st.metric("State Holiday", {'0': 'None', 'a': 'Holiday A', 'b': 'Holiday B', 'c': 'Holiday C'}[state_holiday])
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")

with tab2:
    st.markdown("<h2>About This Model</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='info-box'>
            <h3>📊 Model Details</h3>
            <ul>
                <li><b>Algorithm:</b> XGBoost Regressor</li>
                <li><b>Features:</b> 27 (numeric + encoded categorical)</li>
                <li><b>Training Data:</b> Rossman store sales history</li>
                <li><b>Preprocessing:</b> MinMax Scaling + One-Hot Encoding</li>
            </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='info-box'>
            <h3>🎯 How to Use</h3>
            <ol>
                <li>Select a store ID (1-1115)</li>
                <li>Choose a date for prediction</li>
                <li>Set promotion and holiday flags</li>
                <li>Click "Predict Sales"</li>
                <li>View the estimated sales amount</li>
            </ol>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='info-box'>
        <h3>📝 Features Used</h3>
        <b>Numeric Features:</b> Store, Promo, SchoolHoliday, CompetitionDistance, CompetitionOpen, 
        Promo2, Promo2Open, Day, Month, Year, WeekOfYear, IsPromo2Month
        
        <br><br><b>Categorical Features:</b> DayOfWeek, StateHoliday, StoreType, Assortment
        </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("<h2>📈 Model Performance</h2>", unsafe_allow_html=True)
    
    # Display model metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Model Type",
            "XGBoost",
            delta="Gradient Boosting"
        )
    
    with col2:
        st.metric(
            "Total Features",
            "27",
            delta="After preprocessing"
        )
    
    with col3:
        st.metric(
            "Stores Covered",
            "1115",
            delta="Active stores"
        )
    
    st.markdown("""
        <div class='info-box'>
        <h3>🚀 Model Capabilities</h3>
        <ul>
            <li>✅ Predicts daily sales for any store</li>
            <li>✅ Considers promotions and holidays</li>
            <li>✅ Factors in competition distance</li>
            <li>✅ Accounts for store type and assortment</li>
            <li>✅ Utilizes seasonal patterns</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9rem;'>
    <p>Rossman Sales Predictor | Built with Streamlit & XGBoost | 2024</p>
    </div>
""", unsafe_allow_html=True)
