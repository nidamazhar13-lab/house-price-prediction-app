import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "AmesHousing.csv")

HERO_IMAGE_URL = "https://raw.githubusercontent.com/nidamazhar13-lab/house-price-prediction-app/main/assets/house_bg.png"

# ============================================================
# LOAD MODEL + DATA
# ============================================================

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please keep house_price_model.pkl in the same folder as app.py.")
    st.stop()

model = joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

data = load_data()

# ============================================================
# CSS DESIGN
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(79, 124, 255, 0.16), transparent 35%),
        radial-gradient(circle at bottom right, rgba(139, 92, 246, 0.14), transparent 35%),
        linear-gradient(135deg, #EEF7FF 0%, #F8FBFF 50%, #EAF2FF 100%);
    color: #0F172A;
}

.block-container {
    max-width: 1240px !important;
    padding-top: 1.4rem;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 2rem;
}

header[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.96);
    border-right: 1px solid #E2E8F0;
    box-shadow: 8px 0 30px rgba(15, 23, 42, 0.05);
}

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 12px 0 22px 0;
}

.logo-icon {
    background: linear-gradient(135deg, #4F7CFF, #8B5CF6);
    width: 44px;
    height: 44px;
    border-radius: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 22px;
    box-shadow: 0 10px 24px rgba(79, 124, 255, 0.28);
}

.logo-title {
    font-weight: 900;
    color: #0F172A;
    font-size: 18px;
    line-height: 1.1;
}

.logo-subtitle {
    color: #64748B;
    font-size: 12px;
    font-weight: 700;
}

/* Sidebar radio */
div[role="radiogroup"] label {
    background: transparent;
    padding: 10px 12px;
    border-radius: 14px;
    margin-bottom: 8px;
    font-weight: 700;
    color: #334155;
}

div[role="radiogroup"] label:hover {
    background: #EEF4FF;
}

/* Hero */
.hero {
    background:
        linear-gradient(90deg, rgba(8, 20, 48, 0.88), rgba(8, 20, 48, 0.36)),
        url(""" + HERO_IMAGE_URL + """);
    background-size: cover;
    background-position: center;
    min-height: 285px;
    border-radius: 34px;
    padding: 44px 48px;
    display: flex;
    align-items: center;
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.18);
    border: 1px solid rgba(255,255,255,0.75);
    margin-bottom: 24px;
}

.hero-content {
    max-width: 720px;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.92);
    color: #2563EB;
    padding: 9px 16px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 900;
    margin-bottom: 14px;
}

.hero h1 {
    color: white;
    font-size: 48px;
    font-weight: 900;
    line-height: 1.08;
    letter-spacing: -0.5px;
    margin-bottom: 14px;
}

.hero p {
    color: #EAF2FF;
    font-size: 17px;
    line-height: 1.6;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.94);
    border: 1px solid rgba(226,232,240,0.95);
    border-radius: 28px;
    padding: 26px;
    margin-bottom: 22px;
    box-shadow: 0 16px 42px rgba(15,23,42,0.08);
}

.card h3 {
    color: #0F172A;
    font-weight: 900;
}

.form-card {
    background: rgba(255,255,255,0.96);
    border: 1px solid #E2E8F0;
    border-radius: 30px;
    padding: 28px;
    margin-bottom: 22px;
    box-shadow: 0 18px 45px rgba(15,23,42,0.08);
}

.form-title {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 18px;
}

.form-title h2 {
    color: #0F172A;
    font-weight: 900;
    margin: 0;
}

.form-title p {
    color: #64748B;
    font-weight: 600;
    margin: 6px 0 0 0;
}

.chip {
    background: linear-gradient(135deg, #4F7CFF, #8B5CF6);
    color: white;
    padding: 10px 16px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 13px;
}

/* Tabs */
div[data-testid="stTabs"] button {
    background-color: #F1F5F9;
    color: #0F172A;
    border-radius: 999px;
    font-weight: 800;
    padding: 10px 18px;
    margin-right: 8px;
    border: 1px solid #E2E8F0;
}

div[data-testid="stTabs"] button:hover {
    background: linear-gradient(135deg, #4F7CFF, #8B5CF6);
    color: white;
}

/* Inputs */
[data-testid="stNumberInput"] input {
    background-color: #F8FBFF;
    color: #0F172A;
    border: 1px solid #D8E4F0;
    border-radius: 15px;
    font-weight: 800;
    min-height: 42px;
}

div[data-baseweb="select"] > div {
    background-color: #F8FBFF;
    border: 1px solid #D8E4F0;
    border-radius: 15px;
    color: #0F172A;
    font-weight: 800;
    min-height: 42px;
}

/* Sliders */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #4F7CFF !important;
    border-color: #4F7CFF !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #8B5CF6);
    color: white;
    border: none;
    border-radius: 18px;
    padding: 0.95rem 1rem;
    font-size: 17px;
    font-weight: 900;
    box-shadow: 0 14px 30px rgba(79,124,255,0.25);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1D4ED8, #7C3AED);
    color: white;
}

/* Metric cards */
.metric-card {
    background: white;
    border: 1px solid #E2E8F0;
    padding: 24px;
    border-radius: 24px;
    box-shadow: 0 14px 32px rgba(15,23,42,0.07);
    margin-bottom: 16px;
}

.metric-label {
    color: #64748B;
    font-size: 14px;
    font-weight: 800;
    margin-bottom: 8px;
}

.metric-value {
    color: #0F172A;
    font-size: 28px;
    font-weight: 900;
}

.metric-small {
    color: #22C55E;
    font-size: 13px;
    font-weight: 800;
    margin-top: 4px;
}

/* Prediction */
.prediction-box {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
    padding: 38px;
    border-radius: 30px;
    color: white;
    box-shadow: 0 18px 45px rgba(15,23,42,0.22);
    margin-bottom: 22px;
}

.prediction-box h2 {
    color: #BFD7EA;
    font-size: 20px;
    font-weight: 900;
    margin-bottom: 8px;
}

.prediction-box h1 {
    font-size: 52px;
    font-weight: 900;
    margin: 0;
}

.info-box {
    background: #F8FBFF;
    padding: 18px;
    border-radius: 18px;
    border-left: 6px solid #4F7CFF;
    margin-bottom: 15px;
    color: #0F172A;
}

.footer {
    text-align: center;
    color: #64748B;
    font-size: 14px;
    font-weight: 800;
    margin-top: 28px;
}

/* Mobile */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
    }

    .hero {
        min-height: 240px;
        padding: 26px;
        border-radius: 24px;
        align-items: flex-end;
    }

    .hero h1 {
        font-size: 31px;
    }

    .hero p {
        font-size: 14px;
    }

    .form-card, .card {
        padding: 18px;
        border-radius: 22px;
    }

    .form-title {
        display: block;
    }

    .chip {
        display: inline-block;
        margin-top: 10px;
    }

    .prediction-box h1 {
        font-size: 34px;
    }

    .metric-value {
        font-size: 22px;
    }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">🏡</div>
        <div>
            <div class="logo-title">House Price</div>
            <div class="logo-subtitle">AI Predictor</div>
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
    <div class="card" style="padding:16px; border-radius:20px;">
        <b>Ames Housing Project</b><br>
        <span style="color:#64748B; font-size:13px;">Machine Learning Price Estimator</span>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HELPER COMPONENTS
# ============================================================

def hero_section():
    st.markdown("""
    <div class="hero">
        <div class="hero-content">
            <div class="hero-badge">AI Real Estate Estimator</div>
            <h1>Find Your Home’s Estimated Value</h1>
            <p>
            Predict house selling prices using property area, quality, rooms,
            location, garage, basement and age — powered by machine learning.
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


# ============================================================
# DASHBOARD PAGE
# ============================================================

def dashboard_page():
    hero_section()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Dataset", "Ames", "Housing data")
    with c2:
        metric_card("Prediction Type", "Regression", "SalePrice target")
    with c3:
        metric_card("Input Features", "16", "Area, rooms, location")
    with c4:
        metric_card("Interface", "Responsive", "Mobile + laptop")

    st.markdown("""
    <div class="card">
        <h3>Project Overview</h3>
        <p>
        This dashboard predicts house selling price using machine learning.
        The user can enter property details, view dataset insights,
        explore graphs, and understand important price factors.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PREDICT PRICE PAGE
# ============================================================

def predict_page():
    hero_section()

    st.markdown("""
    <div class="form-card">
        <div class="form-title">
            <div>
                <h2>🏠 House Price Prediction Form</h2>
                <p>Fill the house details below. Results will appear only after clicking Predict.</p>
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
                    max_value=5642,
                    value=0,
                    step=50
                )

                lot_area = st.number_input(
                    "Lot Area (sq ft)",
                    min_value=0,
                    max_value=215245,
                    value=0,
                    step=100
                )

            with c2:
                bedrooms = st.slider("Bedrooms", 0, 8, 0)
                total_rooms = st.slider("Total Rooms Above Ground", 0, 15, 0)

            with c3:
                full_bath = st.slider("Full Bathrooms", 0, 4, 0)
                half_bath = st.slider("Half Bathrooms", 0, 2, 0)

        with tab2:
            c1, c2, c3 = st.columns(3)

            with c1:
                overall_qual = st.slider("Overall Quality", 0, 10, 0)
                overall_cond = st.slider("Overall Condition", 0, 9, 0)

            with c2:
                total_bsmt_sf = st.number_input(
                    "Total Basement Area (sq ft)",
                    min_value=0,
                    max_value=6110,
                    value=0,
                    step=50
                )

                garage_area = st.number_input(
                    "Garage Area (sq ft)",
                    min_value=0,
                    max_value=1488,
                    value=0,
                    step=50
                )

            with c3:
                garage_cars = st.slider("Garage Car Capacity", 0, 5, 0)
                house_age = st.slider("House Age", 0, 136, 0)
                remod_age = st.slider("Years Since Remodel", 0, 60, 0)

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

        submit = st.form_submit_button("Predict House Price", use_container_width=True)

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
            st.warning("Please fill the required fields: living area, lot area, total rooms, quality, condition, location, building type, and house style.")
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
            metric_card("Living Area", f"{gr_liv_area:,} sq ft", "Input value")
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

        fig = px.bar(chart_data, x="Feature", y="Value", text="Value", title="Input Feature Visualization")
        fig.update_traces(marker_color="#4F7CFF", textfont_color="#0F172A")
        fig.update_layout(
            height=440,
            paper_bgcolor="rgba(255,255,255,0.96)",
            plot_bgcolor="rgba(248,251,255,0.96)",
            font=dict(color="#0F172A"),
            title_font=dict(color="#0F172A", size=22),
            xaxis=dict(gridcolor="rgba(15,23,42,0.08)"),
            yaxis=dict(gridcolor="rgba(15,23,42,0.08)")
        )

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# DATA INSIGHTS PAGE
# ============================================================

def data_insights_page():
    hero_section()

    st.markdown("""
    <div class="card">
        <h3>📊 Data Insights</h3>
        <p>
        The Ames Housing dataset contains property features such as area, rooms,
        bathrooms, neighborhood, garage, basement, quality, condition, and selling price.
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
            missing_count = int(data.isnull().sum().sum())
            metric_card("Missing Values", f"{missing_count:,}", "Before handling")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Dataset Preview")
        st.dataframe(data.head(20), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("AmesHousing.csv file not found. Upload it to show dataset insights.")


# ============================================================
# GRAPHS PAGE
# ============================================================

def graphs_page():
    hero_section()

    st.markdown("""
    <div class="card">
        <h3>📈 Graphs & Analysis</h3>
        <p>Visual analysis helps understand which features affect house price.</p>
    </div>
    """, unsafe_allow_html=True)

    if data is None:
        st.warning("AmesHousing.csv file not found. Upload it to show graphs.")
        return

    if "SalePrice" in data.columns:
        fig1 = px.histogram(data, x="SalePrice", nbins=40, title="Sale Price Distribution")
        fig1.update_traces(marker_color="#4F7CFF")
        fig1.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig1, use_container_width=True)

    if "Overall Qual" in data.columns and "SalePrice" in data.columns:
        avg_quality = data.groupby("Overall Qual")["SalePrice"].mean().reset_index()
        fig2 = px.bar(avg_quality, x="Overall Qual", y="SalePrice", title="Average Sale Price by Overall Quality")
        fig2.update_traces(marker_color="#8B5CF6")
        fig2.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig2, use_container_width=True)

    if "Gr Liv Area" in data.columns and "SalePrice" in data.columns:
        fig3 = px.scatter(data, x="Gr Liv Area", y="SalePrice", title="Living Area vs Sale Price")
        fig3.update_traces(marker_color="#2563EB", opacity=0.65)
        fig3.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig3, use_container_width=True)


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

def model_performance_page():
    hero_section()

    st.markdown("""
    <div class="card">
        <h3>🎯 Model Performance</h3>
        <p>
        This section explains how the machine learning model was evaluated.
        Regression models are commonly evaluated using RMSE and MAE.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Evaluation Metric", "RMSE", "Root Mean Squared Error")
    with c2:
        metric_card("Evaluation Metric", "MAE", "Mean Absolute Error")
    with c3:
        metric_card("Target Variable", "SalePrice", "Selling price")

    st.markdown("""
    <div class="card">
        <h3>Model Notes</h3>
        <p>
        The model was trained on selected Ames Housing features. The saved model file
        is loaded through <b>house_price_model.pkl</b> and used to predict estimated selling price.
        </p>
        <p>
        You can add your exact RMSE and MAE values from your notebook in this page if required by your instructor.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ABOUT PAGE
# ============================================================

def about_page():
    hero_section()

    st.markdown("""
    <div class="card">
        <h3>ℹ️ About Project</h3>
        <p>
        <b>Goal:</b> Predict the selling price of houses using machine learning.
        </p>
        <p>
        <b>Dataset:</b> Ames Housing dataset.
        </p>
        <p>
        <b>Process:</b> EDA, missing value handling, feature engineering,
        model training, model evaluation, and Streamlit interface development.
        </p>
        <p>
        <b>Important Features:</b> Living area, lot area, rooms, bathrooms,
        neighborhood, quality, condition, garage, basement, and house age.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Conclusion</h3>
        <p>
        House prices are strongly affected by location, living area, overall quality,
        garage space, basement area, and property age. The final application provides
        a simple and attractive interface where users can enter house details and receive
        an estimated selling price.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE ROUTER
# ============================================================

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
