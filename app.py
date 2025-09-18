import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import folium
from streamlit_folium import st_folium

# Set correct model path
MODEL_PATH = "railway_model_final.tflite"

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("🚉 Railway Track Defect Detection")
st.write("Upload a railway track image and provide GPS location to detect defects.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# GPS input
latitude = st.number_input("Latitude", value=0.0, format="%.6f")
longitude = st.number_input("Longitude", value=0.0, format="%.6f")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(img, caption="Uploaded Image", width=400)

    # Preprocess
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    st.write("🔍 Raw score:", float(prediction))

    # Threshold: 50%
    if prediction < 0.5:
        st.error("⚠️ Defective Track Detected")
    else:
        st.success("✅ Track is Properly Aligned")

    # Show GPS location on map
    st.subheader("📍 Track Location")
    m = folium.Map(location=[latitude, longitude], zoom_start=16)
    folium.Marker([latitude, longitude], tooltip="Track Location").add_to(m)
    st_folium(m, width=700)
