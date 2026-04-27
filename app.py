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
# HELPER FUNCTIONS
# =========================================================
def get_image_base64():
    candidates = [
        os.path.join(BASE_DIR, "assets", "house_bg.jpg"),
        os.path.join(BASE_DIR, "assets", "house_bg.png"),
        os.path.join(BASE_DIR, "house_bg.jpg"),
        os.path.join(BASE_DIR, "house_bg.png"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()
            mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
            return f"data:{mime};base64,{encoded}"
    return None

def hero_section():
    st.markdown("""
    <div class="hero-card">
        <div class="hero-content">
            <div class="hero-badge">Smart House Valuation</div>
            <h1>Predict the Estimated Value of a House</h1>
            <p>
                Enter property features like living area, rooms, bathrooms, location, quality,
                basement, garage and house age to predict an estimated selling price using
                a trained machine learning model.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def metric_card(label, value, subtitle=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

model = load_model()
data = load_data()
hero_image = get_image_base64()

if hero_image:
    hero_background = f"""
        linear-gradient(90deg, rgba(7, 16, 34, 0.88), rgba(7, 16, 34, 0.42)),
        url("{hero_image}")
    """
else:
    hero_background = """
        linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #7C3AED 100%)
    """

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
    color: #0F172A;
}}

.stApp {{
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.10), transparent 30%),
        radial-gradient(circle at bottom right, rgba(124,58,237,0.10), transparent 25%),
        linear-gradient(135deg, #F4F8FF 0%, #EEF4FF 45%, #F8FBFF 100%);
}}

header[data-testid="stHeader"] {{
    background: transparent;
}}

.block-container {{
    max-width: 1250px;
    padding-top: 1rem;
    padding-bottom: 2rem;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #FFFFFF 0%, #F7FAFF 100%) !important;
    border-right: 1px solid #E2E8F0 !important;
    box-shadow: 10px 0 28px rgba(15, 23, 42, 0.06);
}}

[data-testid="stSidebarContent"] {{
    background: transparent !important;
}}

.sidebar-brand {{
    background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 60%, #7C3AED 100%);
    border-radius: 22px;
    padding: 18px 16px;
    color: white;
    margin-bottom: 18px;
    box-shadow: 0 16px 34px rgba(30,58,138,0.20);
}}

.sidebar-brand h2 {{
    margin: 0;
    font-size: 22px;
    font-weight: 900;
    color: white;
}}

.sidebar-brand p {{
    margin: 5px 0 0 0;
    color: #E2E8F0;
    font-size: 13px;
    font-weight: 600;
}}

div[role="radiogroup"] > label {{
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 12px 14px;
    border-radius: 16px;
    margin-bottom: 10px;
    box-shadow: 0 8px 18px rgba(15,23,42,0.04);
}}

div[role="radiogroup"] > label:hover {{
    border-color: #BFDBFE;
    background: #F8FBFF;
}}

.hero-card {{
    background-image: {hero_background};
    background-size: cover;
    background-position: center;
    min-height: 290px;
    border-radius: 30px;
    padding: 38px;
    display: flex;
    align-items: center;
    margin-bottom: 22px;
    box-shadow: 0 22px 50px rgba(15,23,42,0.12);
    border: 1px solid rgba(255,255,255,0.7);
}}

.hero-content {{
    max-width: 760px;
}}

.hero-badge {{
    display: inline-block;
    background: rgba(255,255,255,0.92);
    color: #2563EB;
    padding: 8px 14px;
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
    line-height: 1.08;
}}

.hero-card p {{
    margin: 0;
    color: #EAF2FF;
    font-size: 16px;
    line-height: 1.7;
}}

.glass-card {{
    background: rgba(255,255,255,0.86);
    border: 1px solid rgba(226,232,240,0.95);
    backdrop-filter: blur(10px);
    border-radius: 26px;
    padding: 22px;
    box-shadow: 0 16px 36px rgba(15,23,42,0.06);
    margin-bottom: 20px;
}}

.section-head h2 {{
    margin: 0 0 6px 0;
    font-weight: 900;
    color: #0F172A;
}}

.section-head p {{
    margin: 0;
    color: #64748B;
    font-weight: 600;
}}

.form-chip {{
    display: inline-block;
    background: linear-gradient(135deg, #2563EB, #7C3AED);
    color: white;
    padding: 10px 16px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 900;
}}

.metric-card {{
    background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,251,255,0.96));
    border: 1px solid #E2E8F0;
    border-radius: 22px;
    padding: 18px;
    box-shadow: 0 14px 28px rgba(15,23,42,0.05);
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
    background: linear-gradient(135deg, #0F172A 0%, #1D4ED8 50%, #7C3AED 100%);
    border-radius: 28px;
    padding: 30px;
    color: white;
    box-shadow: 0 20px 40px rgba(37,99,235,0.24);
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
    padding: 14px 16px;
    border-radius: 16px;
    margin-bottom: 14px;
    color: #0F172A;
}}

div[data-baseweb="tab-list"] {{
    gap: 10px;
    margin-bottom: 10px;
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
    background: linear-gradient(135deg, #2563EB, #7C3AED) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 12px 24px rgba(37,99,235,0.22);
}}

[data-testid="stNumberInput"] input {{
    background: #F9FBFF !important;
    border: 1px solid #D9E2F0 !important;
    border-radius: 16px !important;
    color: #0F172A !important;
    font-weight: 800 !important;
    min-height: 44px !important;
}}

div[data-baseweb="select"] > div {{
    background: #F9FBFF !important;
    border: 1px solid #D9E2F0 !important;
    border-radius: 16px !important;
    color: #0F172A !important;
    font-weight: 800 !important;
    min-height: 44px !important;
}}

.stSlider {{
    padding-top: 6px;
    padding-bottom: 8px;
}}

.stSlider [data-baseweb="slider"] {{
    padding-top: 10px !important;
    padding-bottom: 10px !important;
}}

.stSlider [data-baseweb="slider"] > div > div {{
    background: linear-gradient(90deg, #22C55E 0%, #3B82F6 45%, #7C3AED 100%) !important;
    height: 8px !important;
    border-radius: 999px !important;
}}

.stSlider [data-baseweb="slider"] div[role="slider"] {{
    background: #FFFFFF !important;
    border: 4px solid #2563EB !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.25);
    width: 20px !important;
    height: 20px !important;
}}

.stSlider [data-testid="stTickBar"] {{
    display: none !important;
}}

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
    box-shadow: 0 18px 34px rgba(37,99,235,0.24);
    transition: all 0.2s ease;
}}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 22px 38px rgba(37,99,235,0.30);
}}

.stDataFrame {{
    border-radius: 18px !important;
    overflow: hidden !important;
}}

.footer {{
    text-align: center;
    color: #64748B;
    font-weight: 700;
    font-size: 14px;
    margin-top: 16px;
    padding: 8px 0;
}}

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
        padding-left: 0.8rem;
        padding-right: 0.8rem;
        padding-top: 0.7rem;
    }}

    .hero-card {{
        min-height: 220px;
        padding: 20px;
        border-radius: 22px;
        background-position: center;
    }}

    .hero-card h1 {{
        font-size: 28px;
    }}

    .hero-card p {{
        font-size: 14px;
        line-height: 1.55;
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

    .stSlider [data-baseweb="slider"] > div > div {{
        height: 7px !important;
    }}

    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        width: 18px !important;
        height: 18px !important;
    }}
}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>🏡 HomeValue</h2>
        <p>Responsive House Price Prediction App</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔮 Predict Price",
            "📊 Data Insights",
            "📈 Graphs & Analysis",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

# =========================================================
# PAGES
# =========================================================
def dashboard_page():
    hero_section()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Dataset", "Ames", "Housing dataset")
    with c2:
        metric_card("Model", "Regression", "Selling price prediction")
    with c3:
        metric_card("Key Inputs", "16", "Area, rooms, quality, location")
    with c4:
        metric_card("Design", "Responsive", "Mobile + laptop + tablet")

    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>Project Overview</h2>
            <p>
                This application predicts the estimated selling price of a house using machine learning.
                It analyzes inputs like area, rooms, bathrooms, basement, garage, location,
                quality, and house age.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>What This App Does</h2>
            <p>
                • Predict house price<br>
                • Show useful data insights<br>
                • Display attractive graphs and analysis<br>
                • Provide a clean and responsive interface
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def predict_page():
    if model is None:
        st.error("house_price_model.pkl file not found. Keep it in the same folder as app.py.")
        return

    hero_section()

    st.markdown("""
    <div class="glass-card">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;">
            <div class="section-head">
                <h2>Enter House Details</h2>
                <p>Fill the values below and click Predict House Price.</p>
            </div>
            <div class="form-chip">Ames Housing Model</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        tab1, tab2, tab3 = st.tabs(["📐 Size & Rooms", "⭐ Quality & Age", "📍 Location & Type"])

        # ---------------- TAB 1 ----------------
        with tab1:
            col1, col2 = st.columns(2)

            with col1:
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

            with col2:
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

        # ---------------- TAB 2 ----------------
        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                overall_qual = st.slider("Overall Quality", 0, 10, 0)
                overall_cond = st.slider("Overall Condition", 0, 10, 0)
                house_age = st.slider("House Age", 0, 200, 0)

            with col2:
                remod_age = st.slider("Years Since Remodel", 0, 100, 0)
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

        # ---------------- TAB 3 ----------------
        with tab3:
            col1, col2 = st.columns(2)

            neighborhoods = [
                "Select Location", "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst",
                "NridgHt", "Gilbert", "Sawyer", "NWAmes", "SawyerW",
                "BrkSide", "Crawfor", "Mitchel", "NoRidge", "Timber",
                "IDOTRR", "ClearCr", "StoneBr", "SWISU", "MeadowV",
                "Blmngtn", "BrDale", "Veenker", "NPkVill", "Blueste"
            ]

            with col1:
                neighborhood = st.selectbox("Neighborhood / Location", neighborhoods)
                bldg_type = st.selectbox(
                    "Building Type",
                    ["Select Building Type", "1Fam", "TwnhsE", "Twnhs", "Duplex", "2fmCon"]
                )

            with col2:
                house_style = st.selectbox(
                    "House Style",
                    ["Select House Style", "1Story", "2Story", "1.5Fin", "1.5Unf", "SFoyer", "SLvl", "2.5Unf", "2.5Fin"]
                )

        submitted = st.form_submit_button("✨ Predict House Price", use_container_width=True)

    if submitted:
        missing_required = (
            gr_liv_area <= 0 or
            lot_area <= 0 or
            total_rooms <= 0 or
            overall_qual <= 0 or
            overall_cond <= 0 or
            neighborhood == "Select Location" or
            bldg_type == "Select Building Type" or
            house_style == "Select House Style"
        )

        if missing_required:
            st.warning("Please fill all important required values properly before prediction.")
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

def data_insights_page():
    hero_section()

    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>Data Insights</h2>
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

def graphs_page():
    hero_section()

    if data is None:
        st.warning("AmesHousing.csv file not found.")
        return

    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>Graphs & Analysis</h2>
            <p>Visual exploration of important housing features and selling price.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "SalePrice" in data.columns:
        fig1 = px.histogram(data, x="SalePrice", nbins=40, title="Sale Price Distribution")
        fig1.update_traces(marker_color="#2563EB")
        fig1.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#F8FBFF",
            font=dict(color="#0F172A")
        )
        st.plotly_chart(fig1, use_container_width=True)

    if "Overall Qual" in data.columns and "SalePrice" in data.columns:
        avg_quality = data.groupby("Overall Qual")["SalePrice"].mean().reset_index()
        fig2 = px.bar(avg_quality, x="Overall Qual", y="SalePrice", title="Average Sale Price by Overall Quality")
        fig2.update_traces(marker_color="#7C3AED")
        fig2.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#F8FBFF",
            font=dict(color="#0F172A")
        )
        st.plotly_chart(fig2, use_container_width=True)

    if "Gr Liv Area" in data.columns and "SalePrice" in data.columns:
        fig3 = px.scatter(data, x="Gr Liv Area", y="SalePrice", title="Living Area vs Sale Price")
        fig3.update_traces(marker_color="#22C55E", opacity=0.65)
        fig3.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#F8FBFF",
            font=dict(color="#0F172A")
        )
        st.plotly_chart(fig3, use_container_width=True)

def about_page():
    hero_section()

    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>About Project</h2>
            <p>
                This machine learning project predicts the selling price of houses using the
                Ames Housing dataset. The model considers features like area, rooms, bathrooms,
                location, quality, condition, garage, basement, and age.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <h2>Project Workflow</h2>
            <p>
                • Data collection<br>
                • Exploratory Data Analysis (EDA)<br>
                • Missing value handling<br>
                • Feature selection / engineering<br>
                • Model training<br>
                • Evaluation using RMSE / MAE<br>
                • Deployment with Streamlit
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# ROUTER
# =========================================================
if page == "🏠 Dashboard":
    dashboard_page()
elif page == "🔮 Predict Price":
    predict_page()
elif page == "📊 Data Insights":
    data_insights_page()
elif page == "📈 Graphs & Analysis":
    graphs_page()
elif page == "ℹ️ About Project":
    about_page()

st.markdown("""
<div class="footer">
    Developed for House Price Prediction Project
</div>
""", unsafe_allow_html=True)
