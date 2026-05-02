import pandas as pd
import streamlit as st

# Referencing the file you moved to the data folder
try:
    df = pd.read_csv("data/sales.csv")
    st.success("Dataset loaded successfully from /data folder!")
    
    # Quick Check: Show the first few rows
    st.write(df.head())
    
    # Step 3: Basic Trend Detection logic
    if df['Sales'].iloc[-1] > df['Sales'].iloc[0]:
        st.info("Trend: Increasing")
    else:
        st.info("Trend: Decreasing")

except Exception as e:
    st.error(f"Error loading data: {e}")