import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Sales Forecasting System", layout="wide")

# ---------------- TITLE ----------------
st.title("📊 Sales Forecasting & Demand Prediction System")
st.markdown("### 🚀 Smart Insights for Business Growth")

# ---------------- FILE UPLOAD ----------------
st.sidebar.header("📁 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"]
)

# ---------------- INITIAL SCREEN ----------------
if uploaded_file is None:
    st.markdown("---")
    st.info("👈 Please upload a dataset from the sidebar to begin analysis")
    st.markdown("""
    ### 📌 Required Dataset Format:
    - Date  
    - Product  
    - Region  
    - Sales  
    """)
    st.stop()

# ---------------- READ FILE ----------------
file_name = uploaded_file.name

if file_name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)

elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
    df = pd.read_excel(uploaded_file)

else:
    st.error("❌ Unsupported file format")
    st.stop()

# ---------------- VALIDATION ----------------
required_columns = ['Date', 'Product', 'Region', 'Sales']

if not all(col in df.columns for col in required_columns):
    st.error("❌ CSV/Excel must contain: Date, Product, Region, Sales")
    st.stop()

# ---------------- PREPROCESS ----------------
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

st.success("✅ Dataset uploaded successfully!")

# ---------------- FILTERS ----------------
st.sidebar.header("🔎 Smart Filters")

product_filter = st.sidebar.selectbox("Select Product", df['Product'].unique())

available_regions = df[df['Product'] == product_filter]['Region'].unique()
region_filter = st.sidebar.selectbox("Select Region", available_regions)

filtered_df = df[(df['Product'] == product_filter) & (df['Region'] == region_filter)]

# ---------------- DATA DISPLAY ----------------
st.subheader("📂 Filtered Dataset")

if filtered_df.empty:
    st.warning("⚠️ No data available for this selection")
else:
    st.dataframe(filtered_df)

# ---------------- GRAPH ----------------
st.subheader("📈 Sales Trend")

if not filtered_df.empty:
    monthly_sales = filtered_df.groupby('Month')['Sales'].sum()
    st.line_chart(monthly_sales)
else:
    st.info("No graph available")

# ---------------- TREND ----------------
st.subheader("📊 Trend Analysis")

if not filtered_df.empty and len(filtered_df) > 1:
    monthly_sales = filtered_df.groupby('Month')['Sales'].sum()

    if monthly_sales.std() > monthly_sales.mean() * 0.3:
        trend = "Volatile ⚠️"
    elif monthly_sales.iloc[-1] > monthly_sales.iloc[0]:
        trend = "Increasing 📈"
    else:
        trend = "Decreasing 📉"

    st.info(f"Trend: {trend}")
else:
    st.info("Not enough data to determine trend")

# ---------------- ALERTS ----------------
st.subheader("🚨 Smart Alerts")

if not filtered_df.empty and len(filtered_df) > 1:
    if monthly_sales.iloc[-1] < monthly_sales.iloc[-2] * 0.5:
        st.error("⚠️ Sales dropped significantly!")
    elif monthly_sales.iloc[-1] > monthly_sales.iloc[-2] * 1.5:
        st.success("🚀 Sales spike detected!")

# ---------------- INSIGHTS ----------------
st.subheader("💡 Product Insights")

if not filtered_df.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", int(filtered_df['Sales'].sum()))
    col2.metric("Average Sales", int(filtered_df['Sales'].mean()))
    col3.metric("Max Sales", int(filtered_df['Sales'].max()))
else:
    st.info("No insights available")

# ---------------- ML MODEL ----------------
df_model = df.copy()
df_model['Product'] = df_model['Product'].astype('category').cat.codes
df_model['Region'] = df_model['Region'].astype('category').cat.codes

X = df_model[['Month', 'Product', 'Region']]
y = df_model['Sales']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
score = r2_score(y, y_pred)

st.subheader("📏 Model Performance")
st.success(f"R² Score: {round(score, 2)}")

# ---------------- CONFIDENCE ----------------
if score > 0.7:
    confidence = "High ✅"
elif score > 0.4:
    confidence = "Medium ⚠️"
else:
    confidence = "Low ❌"

st.info(f"Prediction Confidence: {confidence}")

# ---------------- PREDICTION ----------------
st.subheader("🔮 Predict Future Sales")

col1, col2, col3 = st.columns(3)

month = col1.slider("Month", 1, 12, 6)
product = col2.selectbox("Product", df['Product'].unique())

available_regions_pred = df[df['Product'] == product]['Region'].unique()
region = col3.selectbox("Region", available_regions_pred)

product_code = df_model[df['Product'] == product]['Product'].iloc[0]
region_code = df_model[df['Region'] == region]['Region'].iloc[0]

if st.button("🚀 Predict Sales"):
    prediction = model.predict([[month, product_code, region_code]])
    st.success(f"📈 Predicted Sales: {int(prediction[0])}")

# ---------------- FOOTER ----------------
st.markdown("---")
