import streamlit as st
import ee
import geemap
import pandas as pd

from model import predict_crop


# ------------------
# Earth Engine
# ------------------
credentials = ee.ServiceAccountCredentials(
    st.secrets["earthengine"]["client_email"],
    key_data=st.secrets["earthengine"]["private_key"]
)

ee.Initialize(
    credentials,
    project="cropai-isro-hackathon"
)

st.set_page_config(
    page_title="Crop AI",
    layout="wide"
)


st.title("🌾 Crop AI Monitoring System")

st.write(
"Satellite based crop health monitoring using AI/ML"
)


# ------------------
# Farmer details
# ------------------

st.sidebar.header("👨‍🌾 Farmer Details")

name = st.sidebar.text_input(
    "Farmer Name",
    "Farmer"
)


# ------------------
# Map
# ------------------

st.subheader("🛰 Select Farm Location")


Map = geemap.Map()


Map.add_draw_control()


Map.to_streamlit(
    height=500
)


roi = Map.user_roi



if roi is None:

    st.warning(
        "Please select your farm on map"
    )

    st.stop()



# ------------------
# Satellite Image
# ------------------

collection = (
    ee.ImageCollection(
        "COPERNICUS/S2_SR_HARMONIZED"
    )
    .filterBounds(roi)
    .filterDate(
        "2025-06-01",
        "2025-10-01"
    )
    .filter(
        ee.Filter.lt(
            "CLOUDY_PIXEL_PERCENTAGE",
            20
        )
    )
)


image = collection.median().clip(roi)



# ------------------
# Indices
# ------------------

ndvi = image.normalizedDifference(
    ["B8","B4"]
)


ndwi = image.normalizedDifference(
    ["B3","B8"]
)



ndvi_value = ndvi.reduceRegion(
    ee.Reducer.mean(),
    roi,
    10
).getInfo()["NDVI"]



ndwi_value = ndwi.reduceRegion(
    ee.Reducer.mean(),
    roi,
    10
).getInfo()["NDWI"]



# ------------------
# Temperature
# ------------------

landsat = (
    ee.ImageCollection(
        "LANDSAT/LC08/C02/T1_L2"
    )
    .filterBounds(roi)
    .filterDate(
        "2025-06-01",
        "2025-10-01"
    )
    .median()
)


temperature = (
    landsat
    .select("ST_B10")
    .multiply(0.00341802)
    .add(149)
)



temp_value = temperature.reduceRegion(
    ee.Reducer.mean(),
    roi,
    30
).getInfo()



temp = list(temp_value.values())[0]



# ------------------
# AI Prediction
# ------------------

prediction = predict_crop(
    ndvi_value,
    ndwi_value,
    temp
)



# ------------------
# Dashboard
# ------------------


st.subheader("📊 Farm Analysis")


c1,c2,c3 = st.columns(3)


c1.metric(
    "NDVI",
    round(ndvi_value,3)
)


c2.metric(
    "NDWI",
    round(ndwi_value,3)
)


c3.metric(
    "Temperature",
    round(temp,2)
)



st.subheader("🤖 AI Crop Health")


if prediction=="Healthy":

    st.success(
        "Healthy Crop 🟢"
    )

elif prediction=="Medium":

    st.warning(
        "Moderate Crop 🟡"
    )

else:

    st.error(
        "Poor Crop 🔴"
    )



st.subheader("📌 Recommendation")


if ndwi_value < 0:

    st.write(
    "💧 Water stress detected. Irrigation recommended."
    )

else:

    st.write(
    "✅ Moisture condition is good."
    )


if temp > 40:

    st.write(
    "🌡 High temperature stress detected."
    )


else:

    st.write(
    "🌡 Temperature normal."
    )



st.info(
f"""
Farmer: {name}

AI Prediction: {prediction}

Crop monitoring completed using satellite data.
"""
)
