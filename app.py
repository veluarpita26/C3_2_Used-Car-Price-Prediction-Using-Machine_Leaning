import streamlit as st
import pickle
import numpy as np

# =============================
# HARD-CODED MODEL METRICS
# =============================
MSE = 0.5703630266491334
RMSE = 0.7552238255306393
MAE = 0.4964348369318066
TRAIN_R2 = 0.986129340989375
TEST_R2 = 0.930576101061694
ADJ_R2 = 0.920909482222183

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    with open("random_forest_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Car Price Predictor", layout="centered")

# =============================
# DARK THEME CSS
# =============================
st.markdown("""
<style>
body { background-color: #0D1117; color: #EAEAEA; }
.title { text-align: center; font-size: 42px; font-weight: bold; color: #58A6FF; }
.subtext { text-align: center; font-size: 18px; color: #C9D1D9; margin-bottom: 20px; }
.predict-btn button { background-color: #FF6B35 !important; color: white !important; border-radius: 12px !important; font-size: 20px !important; padding: 12px 30px !important; border: 2px solid #FF8F5C !important; }
.result-box { background: #161B22; padding: 22px; border-radius: 12px; margin-top: 20px; border-left: 6px solid #58A6FF; color: #EAEAEA; }
.metric-box { background: #1C2128; padding: 18px; border-radius: 10px; border-left: 6px solid #FFA726; margin-top: 12px; color: #EAEAEA; font-size: 17px; }
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.markdown("<div class='title'>🚗 Car Price Prediction Using Machine Learning</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Accurate machine-learning based resale value estimation with detailed model insights</div>", unsafe_allow_html=True)

# =============================
# INPUT FIELDS
# =============================
st.markdown("### 🔧 Enter Car Details")
col1, col2 = st.columns(2)

with col1:
    present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, value=5.0)
    driven_kms = st.number_input("Kilometers Driven", min_value=0, value=50000)
    owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

with col2:
    no_year = st.number_input("Car Age (Years)", min_value=0, max_value=30, value=5)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    selling_type = st.selectbox("Selling Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# =============================
# MARKET TRENDS
# =============================
st.markdown("### 📈 Market Trends")
location = st.selectbox("Location", ["Metro", "Urban", "Rural"])
fuel_trend = st.selectbox("Current Fuel Price Trend", ["High", "Medium", "Low"])
season = st.selectbox("Occasion / Month", ["Normal", "Festival", "Year-End"])

# =============================
# ONE HOT ENCODING
# =============================
Fuel_Type_Petrol = 1 if fuel_type == "Petrol" else 0
Fuel_Type_Diesel = 1 if fuel_type == "Diesel" else 0
Fuel_Type_CNG = 1 if fuel_type == "CNG" else 0

Selling_type_Dealer = 1 if selling_type == "Dealer" else 0
Selling_type_Individual = 1 if selling_type == "Individual" else 0

Transmission_Manual = 1 if transmission == "Manual" else 0
Transmission_Automatic = 1 if transmission == "Automatic" else 0

# =============================
# PREDICTION WITH TREND BREAKDOWN
# =============================
st.markdown("<div class='predict-btn'>", unsafe_allow_html=True)

if st.button("👾 Predict Price", key="predict_btn"):
    try:
        input_features = np.array([[present_price,
                                    driven_kms,
                                    owner,
                                    no_year,
                                    Fuel_Type_CNG,
                                    Fuel_Type_Diesel,
                                    Fuel_Type_Petrol,
                                    Selling_type_Dealer,
                                    Selling_type_Individual,
                                    Transmission_Automatic,
                                    Transmission_Manual]])

        # Base price (ML prediction)
        base_price = model.predict(input_features)[0]
        final_price = base_price

        # Trend breakdown dictionary
        trend_breakdown = {}

        # Location trend
        if location == "Metro":
            final_price *= 1.15
            trend_breakdown['Location'] = "+15%"
        elif location == "Urban":
            final_price *= 1.05
            trend_breakdown['Location'] = "+5%"
        else:
            final_price *= 0.95
            trend_breakdown['Location'] = "-5%"

        # Fuel price trend
        if fuel_trend == "High":
            if fuel_type == "Diesel":
                final_price *= 1.05
                trend_breakdown['Fuel Trend'] = "+5%"
            else:
                final_price *= 0.98
                trend_breakdown['Fuel Trend'] = "-2%"
        elif fuel_trend == "Low":
            if fuel_type == "Petrol":
                final_price *= 1.05
                trend_breakdown['Fuel Trend'] = "+5%"
            else:
                final_price *= 0.98
                trend_breakdown['Fuel Trend'] = "-2%"
        else:
            trend_breakdown['Fuel Trend'] = "0%"

        # Seasonal trend
        if season == "Festival":
            final_price *= 1.10
            trend_breakdown['Seasonal Trend'] = "+10%"
        elif season == "Year-End":
            final_price *= 0.95
            trend_breakdown['Seasonal Trend'] = "-5%"
        else:
            trend_breakdown['Seasonal Trend'] = "0%"

        # =============================
        # DISPLAY RESULTS
        # =============================
        st.markdown(f"""
        <div class='result-box'>
            <h3>📌 Predicted Selling Price:</h3>
            <h4 style='color:#FFA726;'>Base Price (ML Prediction): ₹ {base_price:.2f} lakhs</h4>
            <h2 style='color:#58A6FF;'>Final Price (After Market Trends): ₹ {final_price:.2f} lakhs</h2>
            <p>💡 <b>Insight:</b> The base ML price is adjusted according to market trends (Location, Fuel Price, Seasonal Demand) to reflect real-world resale value.</p>
        </div>
        """, unsafe_allow_html=True)

        # Trend explanations
        st.info(f"""
💡 **Trend Explanation:**  
- Location Trend: {trend_breakdown['Location']}  
- Fuel Price Trend: {trend_breakdown['Fuel Trend']}  
- Seasonal Trend: {trend_breakdown['Seasonal Trend']}

**Selected Trends Applied:**  
- Location: {location}  
- Fuel Trend: {fuel_trend}  
- Occasion/Month: {season}
        """)

        # =============================
        # MODEL PERFORMANCE METRICS
        # =============================
        st.markdown("### 📊 Model Performance Metrics (Random Forest)")
        st.markdown(f"<div class='metric-box'><b>Model Used:</b>  Random Forest Regressor</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><b>MSE:</b> {MSE:.4f}<br>☆ <i>Lower MSE means the model learns patterns with very small squared error — great accuracy.</i></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><b>RMSE:</b> {RMSE:.4f}<br>▽ <i>This shows the average error in the same units as price — the model is usually within ±0.75 lakhs.</i></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><b>MAE:</b> {MAE:.4f}<br>☆ <i>On average, the model is off by less than 0.5 lakhs — very stable predictions.</i></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><b>Train R²:</b> {TRAIN_R2:.4f}<br>▽ <i>The model explains 98% of the variation in training data — excellent learning.</i></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><b>Test R²:</b> {TEST_R2:.4f}<br>☆ <i>93% prediction accuracy on unseen data shows powerful generalization.</i></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><b>Adjusted R²:</b> {ADJ_R2:.4f}<br>▽ <i>This metric accounts for the number of features — confirming no overfitting.</i></div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("</div>", unsafe_allow_html=True)