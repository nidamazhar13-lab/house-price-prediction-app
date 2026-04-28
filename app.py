import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
import base64

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="HomeValue | House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "AmesHousing.csv")

# =========================================================
# IMAGE HANDLING
# =========================================================
def get_image_base64():
    image_paths = [
        os.path.join(BASE_DIR, "assets", "house_bg.png"),
        os.path.join(BASE_DIR, "assets", "house_bg.jpg"),
        os.path.join(BASE_DIR, "house_bg.png"),
        os.path.join(BASE_DIR, "house_bg.jpg"),
    ]

    for image_path in image_paths:
        if os.path.exists(image_path):
            with open(image_path, "rb") as img:
                encoded = base64.b64encode(img.read()).decode()
                mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
                return f"data:{mime};base64,{encoded}"

    return None


hero_image = get_image_base64()

if hero_image:
    hero_background = f"""
        linear-gradient(90deg, rgba(8,20,48,0.78), rgba(8,20,48,0.35)),
        url("{hero_image}")
    """
else:
    hero_background = """
        linear-gradient(135deg, #2563EB 0%, #4F46E5 55%, #8B5CF6 100%)
    """

# =========================================================
# LOAD MODEL + DATA
# =========================================================
if not os.path.exists(MODEL_PATH):
    st.error("house_price_model.pkl file not found. Please keep it in the same folder as app.py.")
    st.stop()

model = joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

data = load_data()

# =========================================================
# CSS DESIGN
# =========================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background:
        radial-gradient(circle at top left, rgba(79,124,255,0.13), transparent 30%),
        radial-gradient(circle at bottom right, rgba(139,92,246,0.10), transparent 30%),
        linear-gradient(135deg, #EEF7FF 0%, #F8FBFF 48%, #EAF2FF 100%);
    color: #0F172A;
}}

/* Keep Streamlit top icons visible and properly spaced */
header[data-testid="stHeader"] {{
    background: rgba(238, 247, 255, 0.94) !important;
    backdrop-filter: blur(10px);
    height: 3.2rem !important;
    z-index: 999999 !important;
    border-bottom: 1px solid rgba(226,232,240,0.70);
}}

[data-testid="stToolbar"] {{
    right: 0.75rem !important;
    top: 0.45rem !important;
    z-index: 999999 !important;
}}

[data-testid="stDecoration"] {{
    display: none !important;
}}

#MainMenu {{
    visibility: visible !important;
}}

.block-container {{
    max-width: 1260px !important;
    padding-top: 3.8rem;
    padding-left: 1.4rem;
    padding-right: 1.4rem;
    padding-bottom: 2rem;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #FFFFFF 0%, #F4F8FF 100%) !important;
    border-right: 1px solid #E2E8F0 !important;
    box-shadow: 8px 0 28px rgba(15,23,42,0.07);
}}

[data-testid="stSidebarContent"] {{
    background: transparent !important;
}}

.sidebar-brand {{
    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 55%, #8B5CF6 100%);
    border-radius: 24px;
    padding: 18px 16px;
    color: white;
    margin-bottom: 18px;
    box-shadow: 0 16px 34px rgba(79,70,229,0.25);
}}

.sidebar-brand h2 {{
    margin: 0;
    font-size: 23px;
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
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 12px 14px;
    border-radius: 18px;
    margin-bottom: 10px;
    color: #0F172A !important;
    box-shadow: 0 8px 18px rgba(15,23,42,0.04);
}}

div[role="radiogroup"] > label:hover {{
    background: #EEF4FF;
    border-color: #BFDBFE;
}}

/* Hero image */
.hero-card {{
    background-image: {hero_background};
    background-size: cover;
    background-position: center;
    min-height: 285px;
    border-radius: 30px;
    padding: 38px;
    display: flex;
    align-items: center;
    margin-bottom: 22px;
    box-shadow: 0 22px 50px rgba(15,23,42,0.16);
    border: 1px solid rgba(255,255,255,0.75);
    overflow: hidden;
}}

.hero-content {{
    max-width: 760px;
}}

.hero-badge {{
    display: inline-block;
    background: rgba(255,255,255,0.94);
    color: #2563EB;
    padding: 9px 15px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 900;
    margin-bottom: 14px;
}}

.hero-card h1 {{
    margin: 0 0 14px 0;
    color: white;
    font-size: 44px;
    font-weight: 900;
    line-height: 1.1;
}}

.hero-card p {{
    margin: 0;
    color: #F1F5F9;
    font-size: 16px;
    line-height: 1.65;
}}

/* Cards */
.glass-card {{
    background: linear-gradient(180deg, rgba(255,255,255,0.94), rgba(244,248,255,0.96));
    border: 1px solid #DCE5F2;
    border-radius: 26px;
    padding: 22px;
    box-shadow: 0 16px 36px rgba(15,23,42,0.08);
    margin-bottom: 20px;
    color: #0F172A;
}}

.section-head h2 {{
    margin: 0 0 6px 0;
    font-weight: 900;
    color: #0F172A;
}}

.section-head p {{
    margin: 0;
    color: #475569;
    font-weight: 600;
    line-height: 1.6;
}}

.metric-card {{
    background: linear-gradient(180deg, #FFFFFF 0%, #F4F8FF 100%);
    border: 1px solid #DCE5F2;
    border-radius: 22px;
    padding: 18px;
    box-shadow: 0 14px 30px rgba(15,23,42,0.07);
    margin-bottom: 16px;
}}

.metric-label {{
    font-size: 13px;
    font-weight: 800;
    color: #64748B;
    margin-bottom: 5px;
}}

.metric-value {{
    font-size: 28px;
    font-weight: 900;
    color: #0F172A;
}}

.metric-sub {{
    font-size: 12px;
    font-weight: 700;
    color: #2563EB;
    margin-top: 4px;
}}

.prediction-box {{
    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 55%, #8B5CF6 100%);
    border-radius: 28px;
    padding: 30px;
    color: white;
    box-shadow: 0 20px 38px rgba(79,70,229,0.26);
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
    background: #F8FBFF;
    border-left: 5px solid #2563EB;
    padding: 15px 16px;
    border-radius: 16px;
    color: #0F172A;
    margin-bottom: 15px;
    box-shadow: 0 8px 18px rgba(15,23,42,0.04);
}}

/* Tabs */
div[data-baseweb="tab-list"] {{
    gap: 10px;
    margin-bottom: 12px;
    flex-wrap: wrap;
}}

div[data-baseweb="tab"] {{
    background: #F8FBFF !important;
    border: 1px solid #DCE6F5 !important;
    border-radius: 999px !important;
    color: #0F172A !important;
    font-weight: 800 !important;
    padding: 10px 16px !important;
}}

div[data-baseweb="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg, #2563EB, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 10px 22px rgba(79,70,229,0.24);
}}

/* Inputs */
[data-testid="stNumberInput"] input {{
    background: #FFFFFF !important;
    border: 1px solid #D9E2F0 !important;
    border-radius: 16px !important;
    color: #0F172A !important;
    font-weight: 800 !important;
    min-height: 44px !important;
}}

div[data-baseweb="select"] > div {{
    background: #FFFFFF !important;
    border: 1px solid #D9E2F0 !important;
    border-radius: 16px !important;
    color: #0F172A !important;
    font-weight: 700 !important;
    min-height: 44px !important;
}}

label, .stSelectbox label, .stNumberInput label {{
    color: #0F172A !important;
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
    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 50%, #8B5CF6 100%) !important;
    box-shadow: 0 16px 30px rgba(79,70,229,0.26);
    transition: all 0.2s ease;
}}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 20px 34px rgba(79,70,229,0.34);
}}

/* Graph */
.js-plotly-plot, .plotly, .stPlotlyChart {{
    background: transparent !important;
}}

.footer {{
    text-align: center;
    color: #64748B;
    font-weight: 700;
    font-size: 14px;
    margin-top: 16px;
    padding: 8px 0;
}}

/* Mobile */
@media (max-width: 768px) {{
    header[data-testid="stHeader"] {{
        height: 3.1rem !important;
    }}

    [data-testid="stToolbar"] {{
        right: 0.4rem !important;
        top: 0.4rem !important;
    }}

    .block-container {{
        padding-top: 3.6rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
        padding-bottom: 1.5rem;
    }}

    .hero-card {{
        min-height: 220px;
        padding: 20px;
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
}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def hero_section():
    st.markdown("""
    <div class="hero-card">
        <div class="hero-content">
            <div class="hero-badge">Smart Home Valuation</div>
            <h1>Find Your Home’s Estimated Value</h1>
            <p>
                Predict house selling prices using area, rooms, bathrooms, location,
                quality, basement, garage and house age — powered by machine learning.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def metric_card(label, value, small=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{small}</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>🏡 HomeValue</h2>
        <p>House Price Prediction</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔮 Predict Price",
            "📊 Data Insights",
            "📈 Graphs & Analysis",
            "🎯 Model Performance",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

# =========================================================
# DASHBOARD PAGE
# =========================================================
def dashboard_page():
    hero_section()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Dataset", "Ames", "Housing data")

    with c2:
        metric_card("Model Type", "Regression", "Price prediction")

    with c3:
        metric_card("Input Features", "16", "Area, rooms, location")

    with c4:
        metric_card("Interface", "Responsive", "Mobile + laptop + tablet")

    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>Project Overview</h2>
            <p>
                This application predicts the selling price of a house using machine learning.
                Users can enter property details, explore data insights, see graphs,
                and understand the important factors affecting house prices.
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
# PREDICT PAGE
# =========================================================
def predict_page():
    hero_section()

    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>🏠 House Price Prediction Form</h2>
            <p>Fill all important details below and click the button to get the estimated selling price.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        tab1, tab2, tab3 = st.tabs(["📐 Size & Rooms", "⭐ Quality & Age", "📍 Location & Type"])

        with tab1:
            c1, c2 = st.columns(2)

            with c1:
                gr_liv_area = st.number_input(
                    "Living Area / Gr Liv Area (sq ft)",
                    min_value=0,
                    max_value=7000,
                    value=0,
                    step=50
                )

                lot_area = st.number_input(
                    "Lot Area (sq ft)",
                    min_value=0,
                    max_value=250000,
                    value=0,
                    step=100
                )

                bedrooms = st.number_input(
                    "Bedrooms",
                    min_value=0,
                    max_value=20,
                    value=0,
                    step=1
                )

            with c2:
                total_rooms = st.number_input(
                    "Total Rooms Above Ground",
                    min_value=0,
                    max_value=30,
                    value=0,
                    step=1
                )

                full_bath = st.number_input(
                    "Full Bathrooms",
                    min_value=0,
                    max_value=15,
                    value=0,
                    step=1
                )

                half_bath = st.number_input(
                    "Half Bathrooms",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1
                )

        with tab2:
            c1, c2 = st.columns(2)

            with c1:
                overall_qual = st.number_input(
                    "Overall Quality",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1
                )

                overall_cond = st.number_input(
                    "Overall Condition",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1
                )

                house_age = st.number_input(
                    "House Age",
                    min_value=0,
                    max_value=200,
                    value=0,
                    step=1
                )

            with c2:
                remod_age = st.number_input(
                    "Years Since Remodel",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=1
                )

                total_bsmt_sf = st.number_input(
                    "Total Basement Area (sq ft)",
                    min_value=0,
                    max_value=7000,
                    value=0,
                    step=50
                )

                garage_area = st.number_input(
                    "Garage Area (sq ft)",
                    min_value=0,
                    max_value=1800,
                    value=0,
                    step=25
                )

                garage_cars = st.number_input(
                    "Garage Car Capacity",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1
                )

        with tab3:
            c1, c2, c3 = st.columns(3)

            with c1:
                neighborhood = st.selectbox(
                    "Neighborhood / Location",
                    [
                        "Select Location", "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst",
                        "NridgHt", "Gilbert", "Sawyer", "NWAmes", "SawyerW",
                        "BrkSide", "Crawfor", "Mitchel", "NoRidge", "Timber",
                        "IDOTRR", "ClearCr", "StoneBr", "SWISU", "MeadowV",
                        "Blmngtn", "BrDale", "Veenker", "NPkVill", "Blueste"
                    ]
                )

            with c2:
                bldg_type = st.selectbox(
                    "Building Type",
                    ["Select Building Type", "1Fam", "TwnhsE", "Twnhs", "Duplex", "2fmCon"]
                )

            with c3:
                house_style = st.selectbox(
                    "House Style",
                    ["Select House Style", "1Story", "2Story", "1.5Fin", "1.5Unf", "SFoyer", "SLvl", "2.5Unf", "2.5Fin"]
                )

        submit = st.form_submit_button("✨ Predict House Price", use_container_width=True)

    if submit:
        required_missing = (
            gr_liv_area <= 0 or
            lot_area <= 0 or
            total_rooms <= 0 or
            overall_qual <= 0 or
            overall_cond <= 0 or
            neighborhood == "Select Location" or
            bldg_type == "Select Building Type" or
            house_style == "Select House Style"
        )

        if required_missing:
            st.warning("Please fill the required fields properly before prediction.")
            return

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

        try:
            prediction = model.predict(input_data)[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return

        st.markdown(f"""
        <div class="prediction-box">
            <h2>Estimated Selling Price</h2>
            <h1>${prediction:,.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("Living Area", f"{gr_liv_area:,} sq ft", "Entered value")

        with c2:
            metric_card("Total Rooms", total_rooms, "Above ground")

        with c3:
            metric_card("Quality Score", f"{overall_qual}/10", "Overall quality")

        with c4:
            metric_card("House Age", f"{house_age} yrs", "Age of house")

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-head">
            <h2>Selected House Features</h2>
            <p>These are the values used for prediction.</p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(input_data, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        chart_data = pd.DataFrame({
            "Feature": [
                "Living Area", "Lot Area", "Bedrooms", "Full Bath", "Half Bath",
                "Total Rooms", "Quality", "Condition", "Basement Area",
                "Garage Area", "Garage Cars", "House Age", "Remodel Age"
            ],
            "Value": [
                gr_liv_area, lot_area, bedrooms, full_bath, half_bath,
                total_rooms, overall_qual, overall_cond, total_bsmt_sf,
                garage_area, garage_cars, house_age, remod_age
            ]
        })

        fig = px.bar(
            chart_data,
            x="Feature",
            y="Value",
            text="Value",
            title="Input Feature Visualization"
        )

        fig.update_traces(
            marker_color="#3B82F6",
            textposition="outside",
            textfont=dict(color="#0F172A", size=12)
        )

        fig.update_layout(
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#F8FBFF",
            title_font=dict(size=22, color="#0F172A"),
            font=dict(color="#0F172A"),
            xaxis=dict(gridcolor="rgba(15,23,42,0.06)", tickangle=-20),
            yaxis=dict(gridcolor="rgba(15,23,42,0.06)")
        )

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# DATA INSIGHTS PAGE
# =========================================================
def data_insights_page():
    hero_section()

    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>📊 Data Insights</h2>
            <p>A quick overview of the Ames Housing dataset used for this project.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if data is None:
        st.warning("AmesHousing.csv file not found.")
        return

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("Rows", f"{data.shape[0]:,}", "Total records")

    with c2:
        metric_card("Columns", data.shape[1], "Total dataset columns")

    with c3:
        metric_card("Missing Values", int(data.isnull().sum().sum()), "Before handling")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-head">
        <h2>Dataset Preview</h2>
        <p>First 20 rows of the dataset.</p>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(data.head(20), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# GRAPHS PAGE
# =========================================================
def graphs_page():
    hero_section()

    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>📈 Graphs & Analysis</h2>
            <p>Visual exploration of important housing features and selling price.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if data is None:
        st.warning("AmesHousing.csv file not found.")
        return

    if "SalePrice" in data.columns:
        fig1 = px.histogram(data, x="SalePrice", nbins=40, title="Sale Price Distribution")
        fig1.update_traces(marker_color="#3B82F6")
        fig1.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#F8FBFF",
            font=dict(color="#0F172A"),
            title_font=dict(size=22, color="#0F172A")
        )
        st.plotly_chart(fig1, use_container_width=True)

    if "Overall Qual" in data.columns and "SalePrice" in data.columns:
        avg_quality = data.groupby("Overall Qual")["SalePrice"].mean().reset_index()
        fig2 = px.bar(avg_quality, x="Overall Qual", y="SalePrice", title="Average Sale Price by Overall Quality")
        fig2.update_traces(marker_color="#8B5CF6")
        fig2.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#F8FBFF",
            font=dict(color="#0F172A"),
            title_font=dict(size=22, color="#0F172A")
        )
        st.plotly_chart(fig2, use_container_width=True)

    if "Gr Liv Area" in data.columns and "SalePrice" in data.columns:
        fig3 = px.scatter(data, x="Gr Liv Area", y="SalePrice", title="Living Area vs Sale Price")
        fig3.update_traces(marker_color="#2563EB", opacity=0.65)
        fig3.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#F8FBFF",
            font=dict(color="#0F172A"),
            title_font=dict(size=22, color="#0F172A")
        )
        st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# MODEL PERFORMANCE PAGE
# =========================================================
def model_performance_page():
    hero_section()

    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>🎯 Model Performance</h2>
            <p>The model is evaluated using regression metrics such as RMSE and MAE.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("Metric", "RMSE", "Root Mean Squared Error")

    with c2:
        metric_card("Metric", "MAE", "Mean Absolute Error")

    with c3:
        metric_card("Target", "SalePrice", "Selling price")

    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>Model Notes</h2>
            <p>
                The trained model uses selected Ames Housing features and predicts the estimated selling price.
                You can add exact RMSE and MAE values here if required in your project.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# ABOUT PAGE
# =========================================================
def about_page():
    hero_section()

    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>ℹ️ About Project</h2>
            <p><b>Goal:</b> Predict the selling price of houses using machine learning.</p>
            <p><b>Dataset:</b> Ames Housing dataset.</p>
            <p><b>Process:</b> EDA, missing value handling, feature engineering, model training, evaluation, and interface development.</p>
            <p><b>Important Features:</b> Living area, rooms, bathrooms, neighborhood, quality, basement, garage, and house age.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>Conclusion</h2>
            <p>
                House prices are influenced by multiple factors such as location, living area, overall quality,
                garage space, basement size, and house age. This web application provides an easy and responsive
                interface for predicting house selling price.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PAGE ROUTER
# =========================================================
if page == "🏠 Dashboard":
    dashboard_page()
elif page == "🔮 Predict Price":
    predict_page()
elif page == "📊 Data Insights":
    data_insights_page()
elif page == "📈 Graphs & Analysis":
    graphs_page()
elif page == "🎯 Model Performance":
    model_performance_page()
elif page == "ℹ️ About Project":
    about_page()

st.markdown("""
<div class="footer">
    Developed for House Price Prediction Machine Learning Project
</div>
""", unsafe_allow_html=True)
