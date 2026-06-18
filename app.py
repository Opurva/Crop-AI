import streamlit as st
import ee
import geemap

from model import predict_crop


# ----------------------------
# Page config
# ----------------------------

st.set_page_config(
    page_title="Crop AI Monitoring",
    layout="wide"
)


# ----------------------------
# Earth Engine Authentication
# ----------------------------

try:

    credentials = ee.ServiceAccountCredentials(
        st.secrets["earthengine"]["client_email"],
        key_data=st.secrets["earthengine"]["private_key"]
    )

    ee.Initialize(
        credentials,
        project=st.secrets["earthengine"]["project_id"]
    )

except Exception as e:

    st.error("Earth Engine authentication failed")
    st.stop()



# ----------------------------
# Title
# ----------------------------

st.title("🌾 Crop AI Monitoring System")

st.write(
    "Satellite based crop health monitoring using AI + Google Earth Engine"
)



# ----------------------------
# Farmer details
# ----------------------------

st.sidebar.header("👨‍🌾 Farmer Details")


farmer = st.sidebar.text_input(
    "Farmer Name",
    "Farmer"
)



# ----------------------------
# Map
# ----------------------------

st.subheader("🛰 Select Farm Area")


Map = geemap.Map(
    center=[20.59,78.96],
    zoom=5
)


Map.add_draw_control()


Map.to_streamlit(
    height=500
)



roi = Map.user_roi



if roi is None:

    st.warning(
        "Draw your farm boundary on the map first"
    )

    st.stop()



# ----------------------------
# Satellite Image
# ----------------------------


try:

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


    image = (
        collection
        .median()
        .clip(roi)
    )


except:

    st.error(
        "Satellite image not found"
    )

    st.stop()



# ----------------------------
# NDVI
# ----------------------------

ndvi = image.normalizedDifference(
    ["B8","B4"]
)


ndvi_data = (
    ndvi.reduceRegion(
        ee.Reducer.mean(),
        roi,
        10
    )
    .getInfo()
)


ndvi_value = ndvi_data.get(
    "nd",
    0
)



# ----------------------------
# NDWI
# ----------------------------

ndwi = image.normalizedDifference(
    ["B3","B8"]
)


ndwi_data = (
    ndwi.reduceRegion(
        ee.Reducer.mean(),
        roi,
        10
    )
    .getInfo()
)


ndwi_value = ndwi_data.get(
    "nd",
    0
)



# ----------------------------
# Temperature
# ----------------------------

try:

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


    temp_img = (
        landsat
        .select("ST_B10")
        .multiply(0.00341802)
        .add(149)
    )


    temp_data = (
        temp_img.reduceRegion(
            ee.Reducer.mean(),
            roi,
            30
        )
        .getInfo()
    )


    temperature = (
        list(temp_data.values())[0]
        if temp_data
        else 30
    )


except:

    temperature = 30



# ----------------------------
# AI Prediction
# ----------------------------

prediction = predict_crop(
    float(ndvi_value),
    float(ndwi_value),
    float(temperature)
)



# ----------------------------
# Dashboard
# ----------------------------


st.subheader("📊 Farm Analysis")


a,b,c = st.columns(3)


a.metric(
    "NDVI",
    round(ndvi_value,3)
)


b.metric(
    "NDWI",
    round(ndwi_value,3)
)


c.metric(
    "Temperature",
    round(temperature,2)
)



st.subheader("🤖 AI Crop Health")


if prediction == "Healthy":

    st.success(
        "Healthy Crop 🟢"
    )


elif prediction == "Medium":

    st.warning(
        "Medium Crop 🟡"
    )


else:

    st.error(
        "Poor Crop 🔴"
    )



# ----------------------------
# Recommendation
# ----------------------------


st.subheader("📌 Recommendation")


if ndwi_value < 0:

    st.write(
        "💧 Water stress detected. Irrigation suggested."
    )

else:

    st.write(
        "✅ Water condition looks good."
    )


if temperature > 40:

    st.write(
        "🌡 High temperature stress detected."
    )

else:

    st.write(
        "🌡 Temperature is normal."
    )



st.info(
f"""
Farmer : {farmer}

Crop Status : {prediction}

Analysis generated from Sentinel-2 satellite data.
"""
)
