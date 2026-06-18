import streamlit as st
import ee
import geemap
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from model import predict_crop

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="AI Crop Intelligence System",
    layout="wide"
)

st.title("🌾 AI Crop Intelligence System (Advanced)")

st.write("Satellite + AI powered crop monitoring with multi-index analysis")

# ----------------------------
# AUTH
# ----------------------------
try:
    private_key = st.secrets["earthengine"]["private_key"]

    credentials = ee.ServiceAccountCredentials(
        st.secrets["earthengine"]["client_email"],
        key_data=private_key.replace("\\n", "\n")
    )

    ee.Initialize(
        credentials,
        project=st.secrets["earthengine"]["project_id"]
    )

    st.success("Earth Engine Connected ✅")

except Exception as e:
    st.error("Auth failed")
    st.exception(e)
    st.stop()

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.header("👨‍🌾 Farmer Info")

farmer = st.sidebar.text_input("Farmer Name", "Farmer")
crop_type = st.sidebar.selectbox(
    "Crop Type",
    ["Wheat", "Rice", "Maize", "Sugarcane", "Cotton"]
)

# ----------------------------
# MAP
# ----------------------------
st.subheader("🛰 Draw Farm Boundary")

m = geemap.Map(center=[20.59, 78.96], zoom=5)
m.add_draw_control()
m.to_streamlit(height=500)

roi = m.user_roi

if roi is None:
    st.warning("Draw farm boundary first")
    st.stop()

# ----------------------------
# SENTINEL DATA
# ----------------------------
start_date = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
end_date = datetime.today().strftime("%Y-%m-%d")

collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(roi)
    .filterDate(start_date, end_date)
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
)

image = collection.median().clip(roi)

# ----------------------------
# INDICES (ADVANCED)
# ----------------------------

# NDVI
ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")

# NDWI
ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")

# EVI
evi = image.expression(
    "2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))",
    {
        "NIR": image.select("B8"),
        "RED": image.select("B4"),
        "BLUE": image.select("B2")
    }
).rename("EVI")

# SAVI
savi = image.expression(
    "((NIR - RED) / (NIR + RED + 0.5)) * 1.5",
    {
        "NIR": image.select("B8"),
        "RED": image.select("B4")
    }
).rename("SAVI")

# ----------------------------
# LAND SURFACE TEMP (LST)
# ----------------------------
try:
    landsat = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .median()
    )

    lst = landsat.select("ST_B10")
    lst = lst.multiply(0.00341802).add(149).rename("LST")

except:
    lst = ee.Image.constant(30).rename("LST")

# ----------------------------
# STACK FEATURES
# ----------------------------
stack = ndvi.addBands([ndwi, evi, savi, lst])

# ----------------------------
# REGION STATS
# ----------------------------
stats = stack.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=roi,
    scale=10
).getInfo()

ndvi_val = stats.get("NDVI", 0)
ndwi_val = stats.get("NDWI", 0)
evi_val = stats.get("EVI", 0)
savi_val = stats.get("SAVI", 0)
temp_val = stats.get("LST", 30)

# ----------------------------
# ML PREDICTION
# ----------------------------
prediction = predict_crop(
    float(ndvi_val),
    float(ndwi_val),
    float(temp_val)
)

# ----------------------------
# ADVANCED RISK SCORING
# ----------------------------
water_risk = "High" if ndwi_val < 0 else "Low"
heat_risk = "High" if temp_val > 38 else "Low"
veg_health = "Good" if ndvi_val > 0.5 else "Poor"

# ----------------------------
# DASHBOARD
# ----------------------------
st.subheader("📊 Multi-Spectral Analysis")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("NDVI", round(ndvi_val, 3))
col2.metric("NDWI", round(ndwi_val, 3))
col3.metric("EVI", round(evi_val, 3))
col4.metric("SAVI", round(savi_val, 3))
col5.metric("LST", round(temp_val, 2))

# ----------------------------
# AI OUTPUT
# ----------------------------
st.subheader("🤖 AI Crop Intelligence")

st.write("### Crop Prediction:", prediction)

if prediction == "Healthy":
    st.success("Crop is Healthy 🟢")
elif prediction == "Medium":
    st.warning("Crop is Moderate 🟡")
else:
    st.error("Crop is Stressed 🔴")

# ----------------------------
# RISK ENGINE
# ----------------------------
st.subheader("⚠ Risk Analysis Engine")

st.write("💧 Water Stress:", water_risk)
st.write("🌡 Heat Stress:", heat_risk)
st.write("🌱 Vegetation Health:", veg_health)

# ----------------------------
# SMART RECOMMENDATION ENGINE
# ----------------------------
st.subheader("📌 AI Recommendations")

if ndwi_val < 0:
    st.write("💧 Irrigation needed immediately (Water stress detected)")

if temp_val > 38:
    st.write("🌡 Heat protection required (mulching/shade suggested)")

if ndvi_val < 0.3:
    st.write("🌱 Crop growth is weak, consider fertilizer optimization")

if ndvi_val > 0.6:
    st.write("🌾 Good vegetation health, maintain current practices")

# ----------------------------
# ML FEATURE VECTOR EXPORT
# ----------------------------
st.subheader("📦 ML Feature Vector")

feature_df = pd.DataFrame([{
    "NDVI": ndvi_val,
    "NDWI": ndwi_val,
    "EVI": evi_val,
    "SAVI": savi_val,
    "LST": temp_val,
    "Prediction": prediction
}])

st.dataframe(feature_df)

csv = feature_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇ Download Dataset",
    csv,
    "crop_features.csv",
    "text/csv"
)

# ----------------------------
# FOOTER
# ----------------------------
st.info(f"""
Farmer: {farmer}
Crop: {crop_type}
AI System: Advanced Multi-Spectral + ML Hybrid Model
Generated: {datetime.now()}
""")
