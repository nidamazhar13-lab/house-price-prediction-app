import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide"
)

# ============================================================
# CSS DESIGN
# ============================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(156deg, #f8fafc 0%, #e5e7eb 100%);
}

.header-box {
    background: linear-gradient(156deg, #111827, #374151);
    padding: 35px;
    border-radius: 22px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.18);
}

.header-box h1 {
    font-size: 44px;
    margin-bottom: 8px;
}

.header-box p {
    font-size: 18px;
    color: #d1d5db;
}

.card {
    background-color: light pink;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
    margin-bottom: 22px;
}

.metric-card {
    background: linear-gradient(135deg, #ffffff, #eef2ff);
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
}

.metric-card h3 {
    color: #4b5563;
    font-size: 16px;
    margin-bottom: 5px;
}

.metric-card h2 {
    color: #111827;
    font-size: 28px;
}

.prediction-box {
    background: linear-gradient(135deg, #047857, #10b981);
    padding: 35px;
    border-radius: 22px;
    color: white;
    text-align: center;
    box-shadow: 0px 8px 25px rgba(16,185,129,0.35);
    margin-top: 20px;
}

.prediction-box h2 {
    font-size: 24px;
    margin-bottom: 8px;
}

.prediction-box h1 {
    font-size: 46px;
    margin: 0;
}

.info-box {
    background: #f9fafb;
    padding: 16px;
    border-radius: 14px;
    border-left: 5px solid #2563eb;
    margin-bottom: 14px;
}

.footer {
    text-align: center;
    color: #6b7280;
    margin-top: 30px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)
# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)
# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="header-box">
    <h1>🏡 House Price Prediction System</h1>
    <p>Predict house selling price using Machine Learning and Ames Housing features</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.title("🏠 Enter House Details")
st.sidebar.write("Fill the house information below to predict the selling price.")

gr_liv_area = st.sidebar.number_input(
    "Living Area / Gr Liv Area (sq ft)",
    min_value=1,
    max_value=5642,
    value=1,
    step=1
)

lot_area = st.sidebar.number_input(
    "Lot Area (sq ft)",
    min_value=1,
    max_value=215245,
    value=1,
    step=1
)

bedrooms = st.sidebar.slider(
    "Bedrooms",
    min_value=0,
    max_value=20,
    value=0
)

full_bath = st.sidebar.slider(
    "Full Bathrooms",
    min_value=0,
    max_value=10,
    value=0
)

half_bath = st.sidebar.slider(
    "Half Bathrooms",
    min_value=0,
    max_value=5,
    value=0
)

total_rooms = st.sidebar.slider(
    "Total Rooms Above Ground",
    min_value=2,
    max_value=15,
    value=0
)

overall_qual = st.sidebar.slider(
    "Overall Quality",
    min_value=1,
    max_value=10,
    value=0
)

overall_cond = st.sidebar.slider(
    "Overall Condition",
    min_value=1,
    max_value=10,
    value=0
)

total_bsmt_sf = st.sidebar.number_input(
    "Total Basement Area (sq ft)",
    min_value=0,
    max_value=6000,
    value=900,
    step=0
)

garage_cars = st.sidebar.slider(
    "Garage Car Capacity",
    min_value=0,
    max_value=5,
    value=0
)

garage_area = st.sidebar.number_input(
    "Garage Area (sq ft)",
    min_value=0,
    max_value=1500,
    value=500,
    step=0
)

house_age = st.sidebar.slider(
    "House Age",
    min_value=0,
    max_value=150,
    value=0
)

remod_age = st.sidebar.slider(
    "Years Since Remodel",
    min_value=0,
    max_value=60,
    value=0
)

neighborhood = st.sidebar.selectbox(
    "Neighborhood / Location",
    [
        "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst",
        "NridgHt", "Gilbert", "Sawyer", "NWAmes", "SawyerW",
        "BrkSide", "Crawfor", "Mitchel", "NoRidge", "Timber",
        "IDOTRR", "ClearCr", "StoneBr", "SWISU", "MeadowV",
        "Blmngtn", "BrDale", "Veenker", "NPkVill", "Blueste"
    ]
)

bldg_type = st.sidebar.selectbox(
    "Building Type",
    ["1Fam", "TwnhsE", "Twnhs", "Duplex", "2fmCon"]
)

house_style = st.sidebar.selectbox(
    "House Style",
    ["1Story", "2Story", "1.5Fin", "1.5Unf", "SFoyer", "SLvl", "2.5Unf", "2.5Fin"]
)

# ============================================================
# INPUT DATA FOR MODEL
# Must match training feature names exactly
# ============================================================

input_data = pd.DataFrame({
    "Gr Liv Area": [gr_liv_area],
    "Lot Area": [lot_area],
    "Bedroom AbvGr": [bedrooms],
    "Full Bath": [full_bath],
    "Half Bath": [half_bath],
    "TotRms AbvGrd": [total_rooms],
    "Overall Qual": [overall_qual],
    "Overall Cond": [overall_cond],
    "Total Bsmt SF": [total_bsmt_sf],
    "Garage Cars": [garage_cars],
    "Garage Area": [garage_area],
    "Neighborhood": [neighborhood],
    "Bldg Type": [bldg_type],
    "House Style": [house_style],
    "HouseAge": [house_age],
    "RemodAge": [remod_age]
})

# ============================================================
# TOP METRIC CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Living Area</h3>
        <h2>{gr_liv_area:,} sq ft</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Total Rooms</h3>
        <h2>{total_rooms}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Overall Quality</h3>
        <h2>{overall_qual}/10</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h3>House Age</h3>
        <h2>{house_age} years</h2>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PREDICTION SECTION
# ============================================================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔮 Predicted Selling Price")
st.write("Click the button to predict the estimated house selling price.")

if st.button("Predict House Price", use_container_width=True):
    prediction = model.predict(input_data)[0]

    st.markdown(f"""
    <div class="prediction-box">
        <h2>Estimated Selling Price</h2>
        <h1>${prediction:,.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# SELECTED FEATURES TABLE
# ============================================================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📌 Selected House Features")
st.write("These values are passed to the trained machine learning model.")
st.dataframe(input_data, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FEATURE VISUALIZATION
# ============================================================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Input Feature Visualization")

chart_data = pd.DataFrame({
    "Feature": [
        "Living Area", "Lot Area", "Bedrooms", "Bathrooms",
        "Rooms", "Quality", "Condition", "Basement Area",
        "Garage Area", "Garage Cars", "House Age", "Remodel Age"
    ],
    "Value": [
        gr_liv_area, lot_area, bedrooms, full_bath + half_bath,
        total_rooms, overall_qual, overall_cond, total_bsmt_sf,
        garage_area, garage_cars, house_age, remod_age
    ]
})

fig = px.bar(
    chart_data,
    x="Feature",
    y="Value",
    text="Value",
    title="House Input Feature Values"
)

fig.update_layout(
    height=450,
    xaxis_title="Features",
    yaxis_title="Values"
)

st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# IMPORTANT FEATURES DISCUSSION
# ============================================================

st.markdown("""
<div class="card">
    <h3>⭐ Important Features Affecting House Price</h3>

    <div class="info-box">
        <b>1. Overall Quality:</b> Better material, finishing, and construction quality usually increase house price.
    </div>

    <div class="info-box">
        <b>2. Living Area:</b> Larger houses generally have higher selling prices.
    </div>

    <div class="info-box">
        <b>3. Neighborhood:</b> Location strongly affects house price because demand varies by area.
    </div>

    <div class="info-box">
        <b>4. Garage and Basement:</b> Extra usable space improves property value.
    </div>

    <div class="info-box">
        <b>5. House Age:</b> Newer or recently remodeled houses usually sell at better prices.
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PROJECT SUMMARY
# ============================================================

st.markdown("""
<div class="card">
    <h3>📘 Project Summary</h3>
    <p>
    This project predicts house selling prices using the Ames Housing dataset.
    The machine learning pipeline uses features such as area, rooms, location,
    house age, quality, basement, and garage information.
    </p>
    <p>
    The dataset was explored using EDA, missing values were handled,
    and models such as Linear Regression and tree-based regressors were evaluated
    using RMSE and MAE.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    Developed for House Price Prediction Machine Learning Project
</div>
""", unsafe_allow_html=True)