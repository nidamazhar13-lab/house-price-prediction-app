import os
import base64
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="HomeValue",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PATHS
# =========================================================
DATA_PATH = "AmesHousing.csv"
MODEL_PATH = "house_price_model.pkl"
BG_IMAGE_PATH = os.path.join("assets", "house_bg.jpg")

# =========================================================
# HELPERS
# =========================================================
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None


@st.cache_data(show_spinner=False)
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def get_defaults_from_data(df):
    defaults = {}
    if df is None:
        return defaults

    for col in df.columns:
        if col == "SalePrice":
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            defaults[col] = float(series.median()) if not series.empty else 0
        else:
            mode_vals = df[col].dropna().mode()
            defaults[col] = str(mode_vals.iloc[0]) if not mode_vals.empty else "NA"

    return defaults


def get_model_expected_columns(model):
    fallback_cols = [
        "Gr Liv Area",
        "Lot Area",
        "Bedroom AbvGr",
        "Full Bath",
        "Half Bath",
        "TotRms AbvGrd",
        "Overall Qual",
        "Overall Cond",
        "Year Built",
        "Year Remod/Add",
        "Total Bsmt SF",
        "Garage Cars",
        "Neighborhood",
        "House Style",
        "Bldg Type",
        "Fireplaces"
    ]

    if model is None:
        return fallback_cols

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        for _, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return fallback_cols


def build_input_dataframe(user_input, expected_columns, defaults):
    row = {}
    for col in expected_columns:
        if col in user_input:
            row[col] = user_input[col]
        else:
            row[col] = defaults.get(col, 0)
    return pd.DataFrame([row])


def safe_predict(model, input_df):
    try:
        pred = model.predict(input_df)[0]
        return float(pred)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def extract_feature_importance(model, feature_names=None):
    try:
        estimator = model
        if hasattr(model, "named_steps"):
            # Try to get last estimator from pipeline
            estimator = list(model.named_steps.values())[-1]

        if hasattr(estimator, "feature_importances_"):
            vals = estimator.feature_importances_
            if feature_names is None:
                feature_names = [f"Feature {i+1}" for i in range(len(vals))]
            return pd.DataFrame({
                "Feature": feature_names[:len(vals)],
                "Importance": vals[:len(feature_names)]
            }).sort_values("Importance", ascending=False)

        if hasattr(estimator, "coef_"):
            vals = np.abs(np.ravel(estimator.coef_))
            if feature_names is None:
                feature_names = [f"Feature {i+1}" for i in range(len(vals))]
            return pd.DataFrame({
                "Feature": feature_names[:len(vals)],
                "Importance": vals[:len(feature_names)]
            }).sort_values("Importance", ascending=False)

    except:
        pass

    return None


# =========================================================
# LOAD DATA / MODEL
# =========================================================
df = load_data()
model = load_model()
defaults = get_defaults_from_data(df)
expected_columns = get_model_expected_columns(model)

# =========================================================
# OPTIONS
# =========================================================
if df is not None and "Neighborhood" in df.columns:
    neighborhood_options = sorted(df["Neighborhood"].dropna().astype(str).unique().tolist())
else:
    neighborhood_options = [
        "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "Gilbert",
        "NridgHt", "Sawyer", "NWAmes", "SawyerW", "BrkSide"
    ]

if df is not None and "House Style" in df.columns:
    house_style_options = sorted(df["House Style"].dropna().astype(str).unique().tolist())
else:
    house_style_options = ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer"]

if df is not None and "Bldg Type" in df.columns:
    bldg_type_options = sorted(df["Bldg Type"].dropna().astype(str).unique().tolist())
else:
    bldg_type_options = ["1Fam", "TwnhsE", "Duplex", "Twnhs", "2fmCon"]

bg_base64 = get_base64_image(BG_IMAGE_PATH)

if bg_base64:
    hero_background = f'linear-gradient(90deg, rgba(2,6,23,0.76), rgba(17,24,39,0.45)), url("data:image/jpeg;base64,{bg_base64}")'
else:
    hero_background = 'linear-gradient(135deg, #0F172A 0%, #1E3A8A 55%, #6D28D9 100%)'

# =========================================================
# CSS
# =========================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

body {{
    color: #E5E7EB;
}}

/* App background */
.stApp {{
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.10), transparent 24%),
        radial-gradient(circle at bottom right, rgba(124,58,237,0.10), transparent 24%),
        linear-gradient(135deg, #0B1120 0%, #0F172A 40%, #172554 100%);
    color: #E5E7EB;
}}

/* Hide top bar / share / menu / footer */
header[data-testid="stHeader"] {{
    background: transparent !important;
    height: 0rem !important;
}}

[data-testid="stToolbar"] {{
    display: none !important;
}}

[data-testid="stDecoration"] {{
    display: none !important;
}}

#MainMenu {{
    visibility: hidden !important;
}}

footer {{
    visibility: hidden !important;
}}

.block-container {{
    max-width: 1280px;
    padding-top: 0.9rem;
    padding-bottom: 2rem;
    padding-left: 1rem;
    padding-right: 1rem;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0F172A 0%, #111827 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}}

[data-testid="stSidebarContent"] {{
    background: transparent !important;
}}

.sidebar-brand {{
    background: linear-gradient(135deg, #1D4ED8 0%, #4F46E5 55%, #7C3AED 100%);
    border-radius: 24px;
    padding: 20px 16px;
    color: white;
    margin-bottom: 18px;
    box-shadow: 0 18px 35px rgba(79,70,229,0.28);
}}

.sidebar-brand h2 {{
    margin: 0;
    font-size: 24px;
    font-weight: 900;
    color: white;
}}

.sidebar-brand p {{
    margin: 6px 0 0 0;
    color: #E0E7FF;
    font-size: 13px;
    font-weight: 600;
}}

div[role="radiogroup"] > label {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 12px 14px;
    border-radius: 18px;
    margin-bottom: 10px;
    color: #E5E7EB !important;
}}

div[role="radiogroup"] > label:hover {{
    background: rgba(255,255,255,0.08);
    border-color: rgba(96,165,250,0.35);
}}

div[role="radiogroup"] > label[data-selected="true"],
div[role="radiogroup"] > label:has(input:checked) {{
    background: linear-gradient(135deg, rgba(37,99,235,0.22), rgba(124,58,237,0.22));
    border: 1px solid rgba(96,165,250,0.45);
}}

/* Hero */
.hero-card {{
    background: {hero_background};
    background-size: cover;
    background-position: center;
    min-height: 290px;
    border-radius: 28px;
    padding: 38px;
    display: flex;
    align-items: center;
    margin-bottom: 22px;
    box-shadow: 0 20px 42px rgba(0,0,0,0.28);
    border: 1px solid rgba(255,255,255,0.08);
}}

.hero-content {{
    max-width: 760px;
}}

.hero-badge {{
    display: inline-block;
    background: rgba(255,255,255,0.14);
    color: #DBEAFE;
    padding: 9px 15px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 900;
    margin-bottom: 14px;
    border: 1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(8px);
}}

.hero-card h1 {{
    margin: 0 0 14px 0;
    color: white;
    font-size: 46px;
    font-weight: 900;
    line-height: 1.1;
}}

.hero-card p {{
    margin: 0;
    color: #E5E7EB;
    font-size: 17px;
    line-height: 1.7;
}}

/* General cards */
.glass-card {{
    background: linear-gradient(180deg, rgba(30,41,59,0.88), rgba(17,24,39,0.92));
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border-radius: 24px;
    padding: 22px;
    box-shadow: 0 14px 32px rgba(0,0,0,0.18);
    margin-bottom: 20px;
    color: #E5E7EB;
}}

.section-head h2 {{
    margin: 0 0 6px 0;
    font-weight: 900;
    color: #F8FAFC;
}}

.section-head p {{
    margin: 0;
    color: #CBD5E1;
    font-weight: 500;
}}

.metric-card {{
    background: linear-gradient(180deg, #1E293B 0%, #111827 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 18px;
    box-shadow: 0 12px 24px rgba(0,0,0,0.18);
    margin-bottom: 16px;
}}

.metric-label {{
    font-size: 13px;
    font-weight: 800;
    color: #94A3B8;
    margin-bottom: 5px;
}}

.metric-value {{
    font-size: 28px;
    font-weight: 900;
    color: #FFFFFF;
}}

.metric-sub {{
    font-size: 12px;
    font-weight: 700;
    color: #93C5FD;
    margin-top: 4px;
}}

.prediction-box {{
    background: linear-gradient(135deg, #1D4ED8 0%, #4F46E5 50%, #7C3AED 100%);
    border-radius: 28px;
    padding: 30px;
    color: white;
    box-shadow: 0 20px 38px rgba(79,70,229,0.28);
    margin-bottom: 18px;
}}

.prediction-box h2 {{
    margin: 0 0 8px 0;
    color: #DBEAFE;
    font-size: 19px;
    font-weight: 800;
}}

.prediction-box h1 {{
    margin: 0;
    font-size: 48px;
    font-weight: 900;
}}

.info-box {{
    background: rgba(30,41,59,0.88);
    border-left: 5px solid #3B82F6;
    padding: 14px 16px;
    border-radius: 16px;
    margin-bottom: 14px;
    color: #E5E7EB;
}}

/* Tabs */
div[data-baseweb="tab-list"] {{
    gap: 10px;
    margin-bottom: 10px;
    flex-wrap: wrap;
}}

div[data-baseweb="tab"] {{
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 999px !important;
    color: #E5E7EB !important;
    font-weight: 800 !important;
    padding: 10px 16px !important;
}}

div[data-baseweb="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg, #2563EB, #7C3AED) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 10px 22px rgba(79,70,229,0.24);
}}

/* Inputs */
[data-testid="stNumberInput"] input {{
    background: #0F172A !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 16px !important;
    color: #F8FAFC !important;
    font-weight: 800 !important;
    min-height: 44px !important;
}}

div[data-baseweb="select"] > div {{
    background: #0F172A !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 16px !important;
    color: #F8FAFC !important;
    font-weight: 700 !important;
    min-height: 44px !important;
}}

label, .stSelectbox label, .stNumberInput label {{
    color: #E5E7EB !important;
    font-weight: 700 !important;
}}

/* Button */
.stButton > button,
div[data-testid="stFormSubmitButton"] > button {{
    width: 100%;
    border: none !important;
    min-height: 58px;
    border-radius: 18px !important;
    font-size: 18px !important;
    font-weight: 900 !important;
    color: white !important;
    background: linear-gradient(135deg, #22C55E 0%, #2563EB 48%, #7C3AED 100%) !important;
    box-shadow: 0 16px 30px rgba(37,99,235,0.28);
    transition: all 0.2s ease;
}}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 20px 34px rgba(37,99,235,0.35);
}}

/* Tables and plots */
.stDataFrame, .element-container {{
    border-radius: 18px !important;
}}

.js-plotly-plot, .plotly, .stPlotlyChart {{
    background: transparent !important;
}}

.footer {{
    text-align: center;
    color: #94A3B8;
    font-weight: 700;
    font-size: 14px;
    margin-top: 16px;
    padding: 8px 0;
}}

/* Mobile */
@media (max-width: 992px) {{
    .hero-card {{
        min-height: 250px;
        padding: 24px;
    }}

    .hero-card h1 {{
        font-size: 34px;
    }}

    .hero-card p {{
        font-size: 15px;
    }}

    .metric-value {{
        font-size: 24px;
    }}
}}

@media (max-width: 768px) {{
    .block-container {{
        padding-left: 0.7rem;
        padding-right: 0.7rem;
        padding-top: 0.5rem;
    }}

    .hero-card {{
        min-height: 210px;
        padding: 18px;
        border-radius: 22px;
        background-position: center;
    }}

    .hero-card h1 {{
        font-size: 28px;
        line-height: 1.15;
    }}

    .hero-card p {{
        font-size: 14px;
        line-height: 1.55;
    }}

    .hero-badge {{
        font-size: 11px;
        padding: 8px 12px;
    }}

    .glass-card {{
        padding: 16px;
        border-radius: 20px;
    }}

    .prediction-box {{
        padding: 22px;
        border-radius: 22px;
    }}

    .prediction-box h1 {{
        font-size: 34px;
    }}

    div[data-baseweb="tab"] {{
        padding: 8px 12px !important;
        font-size: 12px !important;
    }}

    .metric-card {{
        padding: 16px;
    }}
}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("""
<div class="sidebar-brand">
    <h2>🏠 HomeValue</h2>
    <p>Responsive House Price Prediction App</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Predict Price", "Data Insights", "Graphs & Analysis", "About Project"],
    label_visibility="collapsed"
)

# =========================================================
# HERO COMMON
# =========================================================
def show_hero():
    st.markdown("""
    <div class="hero-card">
        <div class="hero-content">
            <div class="hero-badge">Smart House Valuation</div>
            <h1>Predict the Estimated Value of a House</h1>
            <p>
                Enter property features like living area, rooms, bathrooms, location,
                quality, basement, garage, and house age to predict an estimated
                selling price using a trained machine learning model.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":
    show_hero()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Dataset</div>
            <div class="metric-value">Ames</div>
            <div class="metric-sub">Housing dataset</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value">Regression</div>
            <div class="metric-sub">Selling price prediction</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Key Inputs</div>
            <div class="metric-value">16</div>
            <div class="metric-sub">Area, rooms, quality, location</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Design</div>
            <div class="metric-value">Responsive</div>
            <div class="metric-sub">Mobile + laptop + tablet</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>Project Overview</h2>
            <p>
                This application predicts the estimated selling price of a house using machine learning.
                It analyzes inputs like area, rooms, bathrooms, basement, garage, location, quality, and house age.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>Workflow</h2>
            <p>
                Data collection → EDA → missing value handling → feature selection →
                model training → evaluation using RMSE/MAE → deployment with Streamlit.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# PREDICT PRICE
# =========================================================
elif page == "Predict Price":
    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>Enter House Details</h2>
            <p>Fill the values below and click <b>Predict House Price</b>.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📐 Size & Rooms", "⭐ Quality & Age", "📍 Location & Type"])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            gr_liv_area = st.number_input("Living Area / Gr Liv Area (sq ft)", min_value=0, max_value=10000, value=0, step=50)
            lot_area = st.number_input("Lot Area (sq ft)", min_value=0, max_value=200000, value=0, step=100)

        with col2:
            bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=0, step=1)
            total_rooms = st.number_input("Total Rooms Above Ground", min_value=0, max_value=25, value=0, step=1)

        with col3:
            full_bath = st.number_input("Full Bathrooms", min_value=0, max_value=15, value=0, step=1)
            half_bath = st.number_input("Half Bathrooms", min_value=0, max_value=10, value=0, step=1)

    with tab2:
        col1, col2, col3 = st.columns(3)

        with col1:
            overall_qual = st.number_input("Overall Quality (0-10)", min_value=0, max_value=10, value=0, step=1)
            overall_cond = st.number_input("Overall Condition (0-10)", min_value=0, max_value=10, value=0, step=1)

        with col2:
            year_built = st.number_input("Year Built", min_value=1800, max_value=datetime.now().year, value=2000, step=1)
            year_remod = st.number_input("Year Remod/Add", min_value=1800, max_value=datetime.now().year, value=2000, step=1)

        with col3:
            total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=10000, value=0, step=50)
            garage_cars = st.number_input("Garage Cars", min_value=0, max_value=10, value=0, step=1)
            fireplaces = st.number_input("Fireplaces", min_value=0, max_value=10, value=0, step=1)

    with tab3:
        col1, col2, col3 = st.columns(3)

        with col1:
            neighborhood = st.selectbox("Neighborhood", neighborhood_options)

        with col2:
            house_style = st.selectbox("House Style", house_style_options)

        with col3:
            bldg_type = st.selectbox("Building Type", bldg_type_options)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict House Price"):
        if model is None:
            st.error("house_price_model.pkl file not found.")
        else:
            user_input = {
                "Gr Liv Area": gr_liv_area,
                "Lot Area": lot_area,
                "Bedroom AbvGr": bedrooms,
                "Full Bath": full_bath,
                "Half Bath": half_bath,
                "TotRms AbvGrd": total_rooms,
                "Overall Qual": overall_qual,
                "Overall Cond": overall_cond,
                "Year Built": year_built,
                "Year Remod/Add": year_remod,
                "Total Bsmt SF": total_bsmt_sf,
                "Garage Cars": garage_cars,
                "Neighborhood": neighborhood,
                "House Style": house_style,
                "Bldg Type": bldg_type,
                "Fireplaces": fireplaces
            }

            input_df = build_input_dataframe(user_input, expected_columns, defaults)
            prediction = safe_predict(model, input_df)

            if prediction is not None:
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Selling Price</h2>
                    <h1>${prediction:,.0f}</h1>
                </div>
                """, unsafe_allow_html=True)

                summary_df = pd.DataFrame({
                    "Feature": list(user_input.keys()),
                    "Value": list(user_input.values())
                })

                st.markdown("""
                <div class="glass-card">
                    <div class="section-head">
                        <h2>Input Summary</h2>
                        <p>These are the values used for the prediction.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(summary_df, use_container_width=True)


# =========================================================
# DATA INSIGHTS
# =========================================================
elif page == "Data Insights":
    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>Data Insights</h2>
            <p>Overview of the Ames Housing dataset used in the project.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.error("AmesHousing.csv file not found.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        missing_df = missing_df[missing_df["Missing Values"] > 0].sort_values("Missing Values", ascending=False)

        st.subheader("Missing Values")
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing values found.")

        st.subheader("Numeric Summary")
        st.dataframe(df.describe().T, use_container_width=True)


# =========================================================
# GRAPHS & ANALYSIS
# =========================================================
elif page == "Graphs & Analysis":
    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>Graphs & Analysis</h2>
            <p>Important visual analysis of the housing dataset and model-related insights.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.error("AmesHousing.csv file not found.")
    else:
        if "SalePrice" in df.columns and "Gr Liv Area" in df.columns:
            fig1 = px.scatter(
                df,
                x="Gr Liv Area",
                y="SalePrice",
                title="Living Area vs Sale Price",
                opacity=0.7
            )
            fig1.update_traces(marker=dict(color="#3B82F6", size=8))
            fig1.update_layout(
                height=430,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#111827",
                font=dict(color="#E5E7EB"),
                title_font=dict(size=22, color="#F8FAFC"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)")
            )
            st.plotly_chart(fig1, use_container_width=True)

        if "Neighborhood" in df.columns and "SalePrice" in df.columns:
            top_neigh = (
                df.groupby("Neighborhood")["SalePrice"]
                .median()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )

            fig2 = px.bar(
                top_neigh,
                x="Neighborhood",
                y="SalePrice",
                title="Top 10 Neighborhoods by Median Sale Price",
                text="SalePrice"
            )
            fig2.update_traces(marker_color="#8B5CF6", texttemplate='%{text:.0f}')
            fig2.update_layout(
                height=430,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#111827",
                font=dict(color="#E5E7EB"),
                title_font=dict(size=22, color="#F8FAFC"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickangle=-25),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)")
            )
            st.plotly_chart(fig2, use_container_width=True)

        if "SalePrice" in df.columns:
            numeric_df = df.select_dtypes(include=np.number)
            corr = numeric_df.corr(numeric_only=True)["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False).head(10)
            corr_df = corr.reset_index()
            corr_df.columns = ["Feature", "Correlation"]

            fig3 = px.bar(
                corr_df,
                x="Feature",
                y="Correlation",
                title="Top Features Correlated with Sale Price",
                text="Correlation"
            )
            fig3.update_traces(marker_color="#22C55E", texttemplate='%{text:.2f}')
            fig3.update_layout(
                height=430,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#111827",
                font=dict(color="#E5E7EB"),
                title_font=dict(size=22, color="#F8FAFC"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickangle=-25),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)")
            )
            st.plotly_chart(fig3, use_container_width=True)

        feature_imp = extract_feature_importance(model, expected_columns)
        if feature_imp is not None and not feature_imp.empty:
            top_imp = feature_imp.head(10)

            fig4 = px.bar(
                top_imp,
                x="Feature",
                y="Importance",
                title="Top Model Feature Importance",
                text="Importance"
            )
            fig4.update_traces(marker_color="#F59E0B")
            fig4.update_layout(
                height=430,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#111827",
                font=dict(color="#E5E7EB"),
                title_font=dict(size=22, color="#F8FAFC"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickangle=-25),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)")
            )
            st.plotly_chart(fig4, use_container_width=True)


# =========================================================
# ABOUT PROJECT
# =========================================================
elif page == "About Project":
    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>About This Project</h2>
            <p>
                Goal: Predict the selling price of houses using machine learning.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>What This Project Does</h2>
            <p>
                This project analyzes important house features such as area, number of rooms,
                location, age of house, quality, bathrooms, garage and basement.
                It performs EDA, handles missing values, trains a regression model,
                and evaluates performance using RMSE / MAE.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>Main Features Used</h2>
            <p>
                • Living area<br>
                • Lot area<br>
                • Bedrooms<br>
                • Bathrooms<br>
                • Total rooms<br>
                • Quality and condition<br>
                • Year built / remodel<br>
                • Basement area<br>
                • Garage capacity<br>
                • Neighborhood / house style / building type
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>Evaluation</h2>
            <p>
                The project can be evaluated using:
                <br>• RMSE (Root Mean Squared Error)
                <br>• MAE (Mean Absolute Error)
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div class="footer">Built with Streamlit • House Price Prediction Project</div>', unsafe_allow_html=True)
