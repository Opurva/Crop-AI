import streamlit as st

from model import predict_crop


st.title("🌾 Crop AI Monitoring System")

st.write(
"Satellite based crop health monitoring"
)


ndvi = st.slider(
"NDVI",
0.0,
1.0,
0.7
)

ndwi = st.slider(
"NDWI",
-1.0,
1.0,
0.1
)

temp = st.slider(
"Temperature",
20,
50,
30
)


if st.button("Analyze Farm"):

    result = predict_crop(
        ndvi,
        ndwi,
        temp
    )


    st.success(
        f"Crop Health Prediction: {result}"
    )
