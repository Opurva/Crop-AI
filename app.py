import streamlit as st
import ee
import pandas as pd
from datetime import datetime, timedelta
import geemap.foliumap as geemap
from streamlit_folium import st_folium

from model import predict_crop

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI Crop Intelligence", layout="wide")
st.title("🌾 AI Crop Intelligence System (Advanced)")

# ----------------------------
# EARTH ENGINE AUTH
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
    st.error("Auth Failed")
    st.exception(e)
    st.stop()

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.header("👨‍🌾 Farmer Info")
farmer = st.sidebar.text_input("Farmer Name", "Farmer")
crop_type = st.sidebar.selectbox("Crop Type", ["Wheat", "Rice", "Maize", "Cotton", "Sugarcane"])

# ----------------------------
# MAP (FIXED WORKING VERSION)
# ----------------------------
st.subheader("🛰 Draw Your Farm Boundary")

m = geemap.Map(center=[20.59, 78.96], zoom=5)
m.add_draw_control()

output = st_folium(m, height=500)

roi = None

if output and "all_drawings" in output and len(output["all_drawings"]) > 0:
    roi = geemap.geojson_to_ee(output["all_drawings"][-1]["geometry"])

if roi is None:
    st.warning("👉 Farm boundary draw karo (polygon tool use karo)")
    st.stop()

# ----------------------------
# DATE RANGE
# ----------------------------
end_date = datetime.today()
start_date = end_date - timedelta(days=90)

start = start_date.strftime("%Y-%m-%d")
end = end_date.strftime("%Y-%m-%d")

# ----------------------------
# SENTINEL IMAGE
# ----------------------------
collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(roi)
    .filterDate(start, end)
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
)

image = collection.median().clip(roi)

# ----------------------------
# INDICES
# ----------------------------
ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")

evi = image.expression(
    "2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))",
    {
        "NIR": image.select("B8"),
        "RED": image.select("B4"),
        "BLUE": image.select("B2")
    }
).rename("EVI")

savi = image.expression(
    "((NIR - RED) / (NIR + RED + 0.5)) * 1.5",
    {
        "NIR": image.select("B8"),
        "RED": image.select("B4")
    }
).rename("SAVI")

# ----------------------------
# TEMPERATURE (LST)
# ----------------------------
try:
    landsat = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(roi)
        .filterDate(start, end)
        .median()
    )

    lst = landsat.select("ST_B10")
    lst = lst.multiply(0.00341802).add(149).rename("LST")

except:
    lst = ee.Image.constant(30).rename("LST")

# ----------------------------
# STACK
# ----------------------------
stack = ndvi.addBands([ndwi, evi, savi, lst])

stats = stack.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=roi,
    scale=10
).getInfo()

ndvi_v = stats.get("NDVI", 0)
ndwi_v = stats.get("NDWI", 0)
evi_v = stats.get("EVI", 0)
savi_v = stats.get("SAVI", 0)
lst_v = stats.get("LST", 30)

# ----------------------------
# ML PREDICTION
# ----------------------------
prediction = predict_crop(
    float(ndvi_v),
    float(ndwi_v),
    float(lst_v)
)

# ----------------------------
# RISK ENGINE
# ----------------------------
water_risk = "HIGH" if ndwi_v < 0 else "LOW"
heat_risk = "HIGH" if lst_v > 38 else "LOW"
veg_health = "GOOD" if ndvi_v > 0.5 else "POOR"

# ----------------------------
# DASHBOARD
# ----------------------------
st.subheader("📊 Farm Analytics")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("NDVI", round(ndvi_v, 3))
c2.metric("NDWI", round(ndwi_v, 3))
c3.metric("EVI", round(evi_v, 3))
c4.metric("SAVI", round(savi_v, 3))
c5.metric("LST", round(lst_v, 2))

# ----------------------------
# AI RESULT
# ----------------------------
st.subheader("🤖 AI Prediction")

st.write("Crop Status:", prediction)

if prediction == "Healthy":
    st.success("Healthy Crop 🟢")
elif prediction == "Medium":
    st.warning("Moderate Crop 🟡")
else:
    st.error("Stressed Crop 🔴")

# ----------------------------
# RISK ANALYSIS
# ----------------------------
st.subheader("⚠ Risk Analysis")

st.write("💧 Water Risk:", water_risk)
st.write("🌡 Heat Risk:", heat_risk)
st.write("🌱 Vegetation:", veg_health)

# ----------------------------
# RECOMMENDATION ENGINE
# ----------------------------
st.subheader("📌 Recommendations")

if ndwi_v < 0:
    st.write("💧 Irrigation needed immediately")

if lst_v > 38:
    st.write("🌡 Heat stress protection required")

if ndvi_v < 0.3:
    st.write("🌱 Crop growth weak — fertilizer needed")

if ndvi_v > 0.6:
    st.write("🌾 Crop condition good")

# ----------------------------
# EXPORT DATASET
# ----------------------------
st.subheader("📦 Export Features")

df = pd.DataFrame([{
    "NDVI": ndvi_v,
    "NDWI": ndwi_v,
    "EVI": evi_v,
    "SAVI": savi_v,
    "LST": lst_v,
    "Prediction": prediction
}])

st.dataframe(df)

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇ Download Data",
    csv,
    "crop_ai_data.csv",
    "text/csv"
)

# ----------------------------
# FOOTER
# ----------------------------
st.info(f"""
Farmer: {farmer}
Crop: {crop_type}
System: Advanced AI + Satellite Intelligence
Generated: {datetime.now()}
""")
