import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
import base64

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# BACKGROUND IMAGE FUNCTION
# ============================================================

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BG_IMAGE_PATH = os.path.join(BASE_DIR, "assets", "house_bg.jpg")

if os.path.exists(BG_IMAGE_PATH):
    bg_image = get_base64_image(BG_IMAGE_PATH)
else:
    bg_image = ""

# ============================================================
# CSS DESIGN
# ============================================================

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background-image:
        linear-gradient(rgba(20, 24, 22, 0.72), rgba(20, 24, 22, 0.78)),
        url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #3C312B;
}}

.block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1180px;
}}

.hero {{
    background: linear-gradient(135deg, rgba(60,49,43,0.96), rgba(64,62,45,0.94));
    padding: 42px;
    border-radius: 32px;
    margin-bottom: 28px;
    box-shadow: 0 22px 55px rgba(0,0,0,0.35);
    border: 1px solid rgba(247, 178, 44, 0.45);
    color: #FFF8EA;
    position: relative;
    overflow: hidden;
}}

.hero::before {{
    content: "";
    position: absolute;
    width: 270px;
    height: 270px;
    background: rgba(247, 178, 44, 0.18);
    border-radius: 50%;
    right: -90px;
    top: -90px;
}}

.hero-badge {{
    display: inline-block;
    background: #F7B22C;
    color: #3C312B;
    padding: 9px 16px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 900;
    margin-bottom: 14px;
    box-shadow: 0 8px 20px rgba(247, 178, 44, 0.28);
    position: relative;
    z-index: 2;
}}

.hero h1 {{
    color: #FFF8EA;
    font-size: 50px;
    line-height: 1.08;
    margin-bottom: 12px;
    font-weight: 900;
    position: relative;
    z-index: 2;
}}

.hero p {{
    color: #F7E8C8;
    font-size: 17px;
    max-width: 850px;
    position: relative;
    z-index: 2;
}}

.card {{
    background: rgba(255, 248, 234, 0.94);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 28px;
    border-radius: 28px;
    box-shadow: 0 16px 42px rgba(0,0,0,0.25);
    margin-bottom: 24px;
    border: 1px solid rgba(247, 178, 44, 0.35);
    color: #3C312B;
}}

.metric-card {{
    background: linear-gradient(135deg, rgba(60,49,43,0.97), rgba(64,62,45,0.96));
    padding: 24px;
    border-radius: 26px;
    text-align: center;
    box-shadow: 0 14px 32px rgba(0,0,0,0.28);
    border: 1px solid rgba(247, 178, 44, 0.45);
    margin-bottom: 16px;
}}

.metric-card h3 {{
    color: #F7B22C;
    font-size: 15px;
    margin-bottom: 5px;
    font-weight: 900;
}}

.metric-card h2 {{
    color: #FFF8EA;
    font-size: 28px;
    margin: 0;
    font-weight: 900;
}}

.prediction-box {{
    background: linear-gradient(135deg, #F7B22C 0%, #C57A16 100%);
    padding: 38px;
    border-radius: 30px;
    color: #3C312B;
    text-align: center;
    box-shadow: 0 18px 45px rgba(247, 178, 44, 0.42);
    margin-top: 20px;
    border: 1px solid rgba(255, 248, 234, 0.70);
}}

.prediction-box h2 {{
    font-size: 22px;
    margin-bottom: 8px;
    font-weight: 900;
}}

.prediction-box h1 {{
    font-size: 54px;
    margin: 0;
    font-weight: 900;
}}

.info-box {{
    background: rgba(247, 232, 200, 0.92);
    padding: 18px;
    border-radius: 18px;
    border-left: 7px solid #F7B22C;
    margin-bottom: 15px;
    color: #3C312B;
    box-shadow: 0 8px 18px rgba(60, 49, 43, 0.10);
}}

div[data-testid="stTabs"] button {{
    background-color: rgba(255, 248, 234, 0.78);
    color: #3C312B;
    border-radius: 999px;
    font-weight: 900;
    padding: 10px 18px;
    margin-right: 8px;
}}

div[data-testid="stTabs"] button:hover {{
    background-color: #F7B22C;
    color: #3C312B;
}}

.stButton > button {{
    background: linear-gradient(135deg, #3C312B 0%, #403E2D 100%);
    color: #F7B22C;
    border: 1px solid rgba(247, 178, 44, 0.55);
    border-radius: 22px;
    padding: 0.95rem 1rem;
    font-size: 17px;
    font-weight: 900;
    box-shadow: 0 14px 30px rgba(0,0,0,0.28);
}}

.stButton > button:hover {{
    background: linear-gradient(135deg, #F7B22C 0%, #C57A16 100%);
    color: #3C312B;
    border: 1px solid #F7B22C;
}}

[data-testid="stNumberInput"] input {{
    background-color: #FFF8EA;
    color: #3C312B;
    border: 2px solid #D7C3A5;
    border-radius: 16px;
    font-weight: 700;
}}

[data-testid="stNumberInput"] input:focus {{
    border: 2px solid #F7B22C;
    box-shadow: 0 0 0 2px rgba(247,178,44,0.25);
}}

div[data-baseweb="select"] > div {{
    background-color: #FFF8EA;
    border: 2px solid #D7C3A5;
    border-radius: 16px;
    color: #3C312B;
    font-weight: 700;
}}

.stSlider {{
    padding-top: 8px;
    padding-bottom: 8px;
}}

.footer {{
    text-align: center;
    color: #FFF8EA;
    margin-top: 28px;
    font-size: 14px;
    font-weight: 700;
}}

@media (max-width: 768px) {{
    .stApp {{
        background-attachment: scroll;
    }}

    .block-container {{
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
    }}

    .hero {{
        padding: 26px;
        border-radius: 24px;
    }}

    .hero h1 {{
        font-size: 32px;
    }}

    .hero p {{
        font-size: 15px;
    }}

    .card {{
        padding: 18px;
        border-radius: 22px;
    }}

    .prediction-box h1 {{
        font-size: 36px;
    }}

    .metric-card h2 {{
        font-size: 22px;
    }}
}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)

# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-badge">AI Real Estate Estimator</div>
    <h1>Find the Estimated Value of Your House</h1>
    <p>
    Enter property details such as area, rooms, bathrooms, location, quality,
    garage, basement, and house age to predict the estimated selling price
    using a trained machine learning model.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INPUT SECTION
# ============================================================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🏠 Enter House Details")
st.write("Fill the values below and click **Predict House Price**.")

tab1, tab2, tab3 = st.tabs(["📐 Size & Rooms", "⭐ Quality & Age", "📍 Location & Type"])

with tab1:
    c1, c2 = st.columns(2)

    with c1:
        gr_liv_area = st.number_input(
            "Living Area / Gr Liv Area (sq ft)",
            min_value=334,
            max_value=5642,
            value=1800,
            step=50
        )

        lot_area = st.number_input(
            "Lot Area (sq ft)",
            min_value=1300,
            max_value=215245,
            value=8000,
            step=100
        )

        bedrooms = st.slider(
            "Bedrooms",
            min_value=0,
            max_value=8,
            value=3
        )

    with c2:
        full_bath = st.slider(
            "Full Bathrooms",
            min_value=0,
            max_value=4,
            value=2
        )

        half_bath = st.slider(
            "Half Bathrooms",
            min_value=0,
            max_value=2,
            value=1
        )

        total_rooms = st.slider(
            "Total Rooms Above Ground",
            min_value=2,
            max_value=15,
            value=7
        )

with tab2:
    c1, c2 = st.columns(2)

    with c1:
        overall_qual = st.slider(
            "Overall Quality",
            min_value=1,
            max_value=10,
            value=7
        )

        overall_cond = st.slider(
            "Overall Condition",
            min_value=1,
            max_value=9,
            value=5
        )

        total_bsmt_sf = st.number_input(
            "Total Basement Area (sq ft)",
            min_value=0,
            max_value=6110,
            value=900,
            step=50
        )

    with c2:
        garage_cars = st.slider(
            "Garage Car Capacity",
            min_value=0,
            max_value=5,
            value=2
        )

        garage_area = st.number_input(
            "Garage Area (sq ft)",
            min_value=0,
            max_value=1488,
            value=500,
            step=50
        )

        house_age = st.slider(
            "House Age",
            min_value=0,
            max_value=136,
            value=20
        )

        remod_age = st.slider(
            "Years Since Remodel",
            min_value=0,
            max_value=60,
            value=10
        )

with tab3:
    c1, c2, c3 = st.columns(3)

    with c1:
        neighborhood = st.selectbox(
            "Neighborhood / Location",
            [
                "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst",
                "NridgHt", "Gilbert", "Sawyer", "NWAmes", "SawyerW",
                "BrkSide", "Crawfor", "Mitchel", "NoRidge", "Timber",
                "IDOTRR", "ClearCr", "StoneBr", "SWISU", "MeadowV",
                "Blmngtn", "BrDale", "Veenker", "NPkVill", "Blueste"
            ]
        )

    with c2:
        bldg_type = st.selectbox(
            "Building Type",
            ["1Fam", "TwnhsE", "Twnhs", "Duplex", "2fmCon"]
        )

    with c3:
        house_style = st.selectbox(
            "House Style",
            ["1Story", "2Story", "1.5Fin", "1.5Unf", "SFoyer", "SLvl", "2.5Unf", "2.5Fin"]
        )

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# INPUT DATA FOR MODEL
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
# METRIC CARDS
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
        <h3>Quality Score</h3>
        <h2>{overall_qual}/10</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h3>House Age</h3>
        <h2>{house_age} yrs</h2>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PREDICTION SECTION
# ============================================================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔮 Predicted Selling Price")
st.write("Click the button below to estimate the house selling price.")

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
    height=460,
    template="plotly_white",
    xaxis_title="Features",
    yaxis_title="Values",
    title_font_size=20
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
    The model uses area, rooms, location, house age, quality, basement, and garage information.
    </p>
    <p>
    The dataset was analyzed using EDA, missing values were handled, and machine learning models were
    evaluated using RMSE and MAE.
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
