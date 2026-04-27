import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

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

HERO_IMAGE_URL = "https://raw.githubusercontent.com/nidamazhar13-lab/house-price-prediction-app/main/assets/house_bg.png"

# =========================================================
# LOAD MODEL
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
# CSS
# =========================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

html {{
    scroll-behavior: smooth;
}}

body {{
    color: #0F172A;
}}

.stApp {{
    background:
        radial-gradient(circle at top left, rgba(79, 124, 255, 0.12), transparent 30%),
        radial-gradient(circle at bottom right, rgba(139, 92, 246, 0.10), transparent 30%),
        linear-gradient(135deg, #EEF4FF 0%, #F8FBFF 55%, #EEF2FF 100%);
    color: #0F172A;
}}

header[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

.block-container {{
    max-width: 1260px !important;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    padding-left: 1.8rem;
    padding-right: 1.8rem;
}}

section[data-testid="stSidebar"] {{
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
    box-shadow: 8px 0 28px rgba(15, 23, 42, 0.06);
}}

section[data-testid="stSidebar"] * {{
    color: #0F172A;
}}

[data-testid="stSidebarContent"] {{
    background: #FFFFFF !important;
}}

.sidebar-logo {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 10px 0 20px 0;
}}

.logo-icon {{
    width: 46px;
    height: 46px;
    border-radius: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #4F7CFF, #8B5CF6);
    color: white;
    font-size: 22px;
    box-shadow: 0 10px 22px rgba(79,124,255,0.28);
}}

.logo-title {{
    font-size: 19px;
    font-weight: 900;
    line-height: 1.1;
    color: #0F172A;
}}

.logo-subtitle {{
    font-size: 12px;
    font-weight: 700;
    color: #64748B;
}}

div[role="radiogroup"] > label {{
    background: transparent;
    padding: 10px 12px;
    border-radius: 14px;
    margin-bottom: 8px;
    color: #334155 !important;
    font-weight: 700;
}}

div[role="radiogroup"] > label:hover {{
    background: #EEF4FF;
}}

.hero {{
    background:
        linear-gradient(90deg, rgba(12, 25, 50, 0.82), rgba(12, 25, 50, 0.42)),
        url('{HERO_IMAGE_URL}');
    background-size: cover;
    background-position: center;
    min-height: 290px;
    border-radius: 32px;
    padding: 42px 44px;
    display: flex;
    align-items: center;
    box-shadow: 0 24px 55px rgba(15,23,42,0.14);
    border: 1px solid rgba(255,255,255,0.75);
    margin-bottom: 22px;
    overflow: hidden;
}}

.hero-content {{
    max-width: 720px;
}}

.hero-badge {{
    display: inline-block;
    background: rgba(255,255,255,0.94);
    color: #2563EB;
    padding: 9px 16px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 900;
    margin-bottom: 15px;
}}

.hero h1 {{
    color: #FFFFFF;
    font-size: 46px;
    font-weight: 900;
    line-height: 1.06;
    letter-spacing: -0.5px;
    margin-bottom: 14px;
}}

.hero p {{
    color: #EAF2FF;
    font-size: 17px;
    line-height: 1.6;
    margin: 0;
}}

.card {{
    background: rgba(255,255,255,0.97);
    border: 1px solid #E2E8F0;
    border-radius: 28px;
    padding: 24px;
    box-shadow: 0 16px 42px rgba(15,23,42,0.06);
    margin-bottom: 20px;
}}

.card h3 {{
    margin-top: 0;
    color: #0F172A;
    font-weight: 900;
}}

.form-card {{
    background: rgba(255,255,255,0.98);
    border: 1px solid #DCE5F2;
    border-radius: 30px;
    padding: 26px;
    box-shadow: 0 16px 44px rgba(15,23,42,0.06);
    margin-bottom: 20px;
}}

.form-title {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    margin-bottom: 16px;
}}

.form-title h2 {{
    margin: 0;
    color: #0F172A;
    font-weight: 900;
}}

.form-title p {{
    margin: 6px 0 0 0;
    color: #64748B;
    font-weight: 600;
}}

.chip {{
    display: inline-block;
    background: linear-gradient(135deg, #4F7CFF, #8B5CF6);
    color: #FFFFFF;
    padding: 10px 16px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 900;
    white-space: nowrap;
}}

.metric-card {{
    background: rgba(255,255,255,0.98);
    border: 1px solid #E2E8F0;
    border-radius: 24px;
    padding: 20px;
    box-shadow: 0 14px 30px rgba(15,23,42,0.05);
    margin-bottom: 16px;
}}

.metric-label {{
    color: #64748B;
    font-weight: 800;
    font-size: 14px;
    margin-bottom: 6px;
}}

.metric-value {{
    color: #0F172A;
    font-size: 28px;
    font-weight: 900;
}}

.metric-small {{
    color: #2563EB;
    font-size: 13px;
    font-weight: 700;
    margin-top: 4px;
}}

.prediction-box {{
    background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 55%, #7C3AED 100%);
    color: white;
    padding: 34px;
    border-radius: 30px;
    box-shadow: 0 18px 45px rgba(30,58,138,0.24);
    margin-bottom: 20px;
}}

.prediction-box h2 {{
    margin: 0 0 8px 0;
    color: #DCE9FF;
    font-size: 20px;
    font-weight: 800;
}}

.prediction-box h1 {{
    margin: 0;
    font-size: 50px;
    font-weight: 900;
}}

.info-box {{
    background: #F8FBFF;
    border-left: 5px solid #4F7CFF;
    padding: 16px;
    border-radius: 16px;
    color: #0F172A;
    margin-bottom: 15px;
}}

.footer {{
    text-align: center;
    color: #64748B;
    font-size: 14px;
    font-weight: 700;
    margin-top: 24px;
}}

.stDataFrame {{
    border-radius: 18px !important;
    overflow: hidden;
}}

div[data-baseweb="tab-list"] {{
    gap: 12px;
}}

div[data-baseweb="tab"] {{
    background: #F6F8FC !important;
    border: 1px solid #DCE5F2 !important;
    border-radius: 999px !important;
    color: #0F172A !important;
    font-weight: 800 !important;
    padding: 10px 18px !important;
}}

div[data-baseweb="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg, #4F7CFF, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 10px 24px rgba(79,124,255,0.24);
}}

[data-testid="stNumberInput"] input {{
    background: #F8FBFF !important;
    border: 1px solid #D7E2F0 !important;
    border-radius: 15px !important;
    font-weight: 800 !important;
    color: #0F172A !important;
    min-height: 42px !important;
}}

div[data-baseweb="select"] > div {{
    background: #F8FBFF !important;
    border: 1px solid #D7E2F0 !important;
    border-radius: 15px !important;
    font-weight: 800 !important;
    color: #0F172A !important;
    min-height: 42px !important;
}}

.stSlider {{
    padding-top: 4px;
}}

.stSlider [data-baseweb="slider"] {{
    padding-top: 8px;
    padding-bottom: 8px;
}}

.stSlider [data-baseweb="slider"] > div > div {{
    background: linear-gradient(90deg, #4F7CFF, #8B5CF6) !important;
    height: 6px !important;
    border-radius: 999px !important;
}}

.stSlider [data-baseweb="slider"] div[role="slider"] {{
    background: #ffffff !important;
    border: 4px solid #4F7CFF !important;
    box-shadow: 0 3px 12px rgba(79,124,255,0.28);
}}

.stSlider [data-testid="stTickBar"] {{
    display: none;
}}

.stButton > button,
div[data-testid="stFormSubmitButton"] > button {{
    width: 100%;
    min-height: 56px;
    border: none !important;
    border-radius: 18px !important;
    color: white !important;
    font-size: 18px !important;
    font-weight: 900 !important;
    background: linear-gradient(135deg, #2563EB 0%, #4F7CFF 45%, #8B5CF6 100%) !important;
    box-shadow: 0 16px 30px rgba(79,124,255,0.28);
    transition: all 0.25s ease-in-out;
}}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 18px 34px rgba(79,124,255,0.34);
}}

.stButton > button:focus,
div[data-testid="stFormSubmitButton"] > button:focus {{
    outline: none !important;
    box-shadow: 0 0 0 0.18rem rgba(79,124,255,0.20);
}}

label, .stMarkdown, p, h1, h2, h3, h4, h5 {{
    color: #0F172A;
}}

@media (max-width: 992px) {{
    .hero {{
        min-height: 250px;
        padding: 26px;
        border-radius: 24px;
    }}

    .hero h1 {{
        font-size: 34px;
    }}

    .hero p {{
        font-size: 15px;
    }}

    .form-title {{
        flex-direction: column;
        align-items: flex-start;
    }}
}}

@media (max-width: 768px) {{
    .block-container {{
        padding-top: 0.8rem;
        padding-left: 0.9rem;
        padding-right: 0.9rem;
        padding-bottom: 1.5rem;
    }}

    .stApp {{
        background: linear-gradient(135deg, #EEF4FF 0%, #F8FBFF 55%, #EEF2FF 100%) !important;
    }}

    section[data-testid="stSidebar"] {{
        background: #FFFFFF !important;
    }}

    [data-testid="stSidebarContent"] {{
        background: #FFFFFF !important;
    }}

    .hero {{
        min-height: 230px;
        padding: 22px;
        border-radius: 22px;
        align-items: flex-end;
    }}

    .hero h1 {{
        font-size: 29px;
    }}

    .hero p {{
        font-size: 14px;
    }}

    .form-card, .card {{
        padding: 16px;
        border-radius: 20px;
    }}

    .metric-value {{
        font-size: 22px;
    }}

    .prediction-box {{
        padding: 24px;
        border-radius: 22px;
    }}

    .prediction-box h1 {{
        font-size: 34px;
    }}

    div[data-baseweb="tab"] {{
        padding: 8px 12px !important;
        font-size: 13px !important;
    }}
}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def hero_section():
    st.markdown("""
    <div class="hero">
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
        <div class="metric-small">{small}</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">🏡</div>
        <div>
            <div class="logo-title">HomeValue</div>
            <div class="logo-subtitle">House Price Prediction</div>
        </div>
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

    st.markdown("---")
    st.markdown("""
    <div class="card" style="padding:16px; border-radius:20px; box-shadow:none;">
        <b>Ames Housing Project</b><br>
        <span style="color:#64748B; font-size:13px;">Responsive ML Price Estimator</span>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# DASHBOARD PAGE
# =========================================================
def dashboard_page():
    hero_section()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Dataset", "Ames", "Housing data")
    with c2:
        metric_card("Model Type", "Regression", "House price prediction")
    with c3:
        metric_card("Input Features", "16", "Area, rooms, quality")
    with c4:
        metric_card("Interface", "Responsive", "Mobile + laptop + tablet")

    st.markdown("""
    <div class="card">
        <h3>Project Overview</h3>
        <p>
            This application predicts the selling price of a house using machine learning.
            Users can enter house details, explore data insights, see analysis graphs,
            and understand the important factors affecting house prices.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PREDICT PAGE
# =========================================================
def predict_page():
    hero_section()

    st.markdown("""
    <div class="form-card">
        <div class="form-title">
            <div>
                <h2>🏠 House Price Prediction Form</h2>
                <p>Fill all important details below and click the button to get the estimated selling price.</p>
            </div>
            <div class="chip">Ames Housing Model</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        tab1, tab2, tab3 = st.tabs(["📐 Size & Rooms", "⭐ Quality & Age", "📍 Location & Type"])

        with tab1:
            c1, c2, c3 = st.columns(3)

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

            with c2:
                bedrooms = st.slider("Bedrooms", 0, 20, 0)
                total_rooms = st.slider("Total Rooms Above Ground", 0, 30, 0)

            with c3:
                full_bath = st.slider("Full Bathrooms", 0, 15, 0)
                half_bath = st.slider("Half Bathrooms", 0, 10, 0)

        with tab2:
            c1, c2, c3 = st.columns(3)

            with c1:
                overall_qual = st.slider("Overall Quality", 0, 10, 0)
                overall_cond = st.slider("Overall Condition", 0, 10, 0)

            with c2:
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

            with c3:
                garage_cars = st.slider("Garage Car Capacity", 0, 10, 0)
                house_age = st.slider("House Age", 0, 200, 0)
                remod_age = st.slider("Years Since Remodel", 0, 100, 0)

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

        prediction = model.predict(input_data)[0]

        st.markdown(f"""
        <div class="prediction-box">
            <h2>Estimated Selling Price</h2>
            <h1>${prediction:,.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Living Area", f"{gr_liv_area:,} sq ft", "User input")
        with c2:
            metric_card("Total Rooms", total_rooms, "Above ground")
        with c3:
            metric_card("Quality Score", f"{overall_qual}/10", "Overall quality")
        with c4:
            metric_card("House Age", f"{house_age} yrs", "Property age")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📌 Selected House Features")
        st.dataframe(input_data, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

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
            title="Input Feature Visualization"
        )
        fig.update_traces(marker_color="#4F7CFF", textfont_color="#0F172A")
        fig.update_layout(
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#F8FBFF",
            font=dict(color="#0F172A"),
            title_font=dict(size=22, color="#0F172A"),
            xaxis=dict(gridcolor="rgba(15,23,42,0.06)"),
            yaxis=dict(gridcolor="rgba(15,23,42,0.06)")
        )

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# DATA INSIGHTS PAGE
# =========================================================
def data_insights_page():
    hero_section()

    st.markdown("""
    <div class="card">
        <h3>📊 Data Insights</h3>
        <p>
            The Ames Housing dataset contains house-related features such as area, rooms,
            bathrooms, quality, condition, garage, basement, neighborhood, and selling price.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if data is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Rows", f"{data.shape[0]:,}", "Total records")
        with c2:
            metric_card("Columns", data.shape[1], "Dataset features")
        with c3:
            metric_card("Missing Values", f"{int(data.isnull().sum().sum()):,}", "Before cleaning")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Dataset Preview")
        st.dataframe(data.head(20), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("AmesHousing.csv file not found.")

# =========================================================
# GRAPHS PAGE
# =========================================================
def graphs_page():
    hero_section()

    st.markdown("""
    <div class="card">
        <h3>📈 Graphs & Analysis</h3>
        <p>These graphs help visualize important trends in house prices.</p>
    </div>
    """, unsafe_allow_html=True)

    if data is None:
        st.warning("AmesHousing.csv file not found.")
        return

    if "SalePrice" in data.columns:
        fig1 = px.histogram(data, x="SalePrice", nbins=40, title="Sale Price Distribution")
        fig1.update_traces(marker_color="#4F7CFF")
        fig1.update_layout(
            template="plotly_white",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#F8FBFF"
        )
        st.plotly_chart(fig1, use_container_width=True)

    if "Overall Qual" in data.columns and "SalePrice" in data.columns:
        avg_quality = data.groupby("Overall Qual")["SalePrice"].mean().reset_index()
        fig2 = px.bar(avg_quality, x="Overall Qual", y="SalePrice", title="Average Sale Price by Overall Quality")
        fig2.update_traces(marker_color="#8B5CF6")
        fig2.update_layout(
            template="plotly_white",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#F8FBFF"
        )
        st.plotly_chart(fig2, use_container_width=True)

    if "Gr Liv Area" in data.columns and "SalePrice" in data.columns:
        fig3 = px.scatter(data, x="Gr Liv Area", y="SalePrice", title="Living Area vs Sale Price")
        fig3.update_traces(marker_color="#2563EB", opacity=0.65)
        fig3.update_layout(
            template="plotly_white",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#F8FBFF"
        )
        st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# MODEL PERFORMANCE PAGE
# =========================================================
def model_performance_page():
    hero_section()

    st.markdown("""
    <div class="card">
        <h3>🎯 Model Performance</h3>
        <p>
            The house price model is evaluated using regression metrics such as RMSE and MAE.
        </p>
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
    <div class="card">
        <h3>Model Notes</h3>
        <p>
            The trained model uses selected Ames Housing features and predicts the estimated selling price.
            You can also add your exact RMSE and MAE values here if required in your project.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# ABOUT PAGE
# =========================================================
def about_page():
    hero_section()

    st.markdown("""
    <div class="card">
        <h3>ℹ️ About Project</h3>
        <p><b>Goal:</b> Predict the selling price of houses using machine learning.</p>
        <p><b>Dataset:</b> Ames Housing dataset.</p>
        <p><b>Process:</b> EDA, missing value handling, feature engineering, model training, evaluation, and interface development.</p>
        <p><b>Important Features:</b> Living area, rooms, bathrooms, neighborhood, quality, basement, garage, and house age.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Conclusion</h3>
        <p>
            House prices are influenced by multiple factors such as location, living area, overall quality,
            garage space, basement size, and house age. This web application provides an easy and responsive
            interface for predicting house selling price.
        </p>
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
