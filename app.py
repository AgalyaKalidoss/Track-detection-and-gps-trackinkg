import os
import random
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Railway Track Fault Detector", layout="wide")

# -------- Load TFLite model --------
@st.cache_resource
def load_model():
    if not os.path.exists("railway_model_final.tflite"):
        st.error("❌ Model file 'railway_model_final.tflite' not found. Please upload it.")
        st.stop()
    interpreter = tf.lite.Interpreter(model_path="railway_model_final.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# -------- Prediction function --------
def predict(image: Image.Image):
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    # Assuming output: [prob_defective, prob_non_defective]
    return prediction

# -------- Simulate GPS & Sensor --------
def get_gps_data():
    return {
        "Latitude": round(random.uniform(8.0, 12.0), 6),
        "Longitude": round(random.uniform(76.0, 80.0), 6)
    }

def get_sensor_data():
    return {
        "Vibration": round(random.uniform(0.0, 1.0), 2),
        "Tilt": round(random.uniform(0.0, 5.0), 2)
    }

# -------- UI --------
st.title("🚂 Railway Track Fault Detector with GPS + Sensors")

uploaded_file = st.file_uploader("Upload a railway track image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Track Image", width="stretch")

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing track..."):
            prediction = predict(image)
            gps = get_gps_data()
            sensor = get_sensor_data()

            st.subheader("📍 GPS Location")
            st.json(gps)

            st.subheader("📡 Sensor Data")
            st.json(sensor)

            st.subheader("📊 Prediction Result")
            defective_prob = float(prediction[0])
            non_defective_prob = float(prediction[1]) if len(prediction) > 1 else (1 - defective_prob)

            if defective_prob > non_defective_prob:
                st.error(f"⚠️ Track is **Defective**\n\nConfidence: {defective_prob:.2%}")
            else:
                st.success(f"✅ Track is **Not Defective**\n\nConfidence: {non_defective_prob:.2%}")
