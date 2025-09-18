import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import geocoder  # For getting current GPS location

# =======================
# Load TFLite Model
# =======================
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="railway_model_final.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =======================
# App UI Setup
# =======================
st.set_page_config(page_title="Railway Track Fault Detector", layout="wide")

st.title("🚂 Railway Track Fault Detection Dashboard")
st.markdown("### Detect defects in real-time using sensor data + GPS + AI")

# =======================
# GPS Section
# =======================
with st.sidebar:
    st.header("📍 GPS Location")
    if st.button("📡 Get Current Location"):
        g = geocoder.ip('me')
        if g.ok:
            st.session_state['lat'] = g.latlng[0]
            st.session_state['lon'] = g.latlng[1]
        else:
            st.warning("Could not fetch GPS automatically. Please enter manually.")
    lat = st.text_input("Latitude", value=str(st.session_state.get('lat', "")))
    lon = st.text_input("Longitude", value=str(st.session_state.get('lon', "")))

    # =======================
    # Sensor Inputs
    # =======================
    st.header("📡 Sensor Data")
    ultrasonic = st.slider("Ultrasonic Distance (cm)", 0, 200, 50)
    vibration = st.slider("Vibration Level (Hz)", 0, 100, 30)
    acoustic = st.slider("Acoustic Level (dB)", 0, 120, 60)
    radar = st.slider("Radar Reflection (%)", 0, 100, 50)

# =======================
# Image Upload
# =======================
st.subheader("📷 Upload Track Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# =======================
# Prediction
# =======================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Track Image", width=400)

    if st.button("🔍 Analyze Track"):
        img_array = preprocess_image(image)
        prediction = predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Combine sensor data and ML prediction
        risk_score = (
            (200 - ultrasonic) * 0.2 +
            vibration * 0.2 +
            acoustic * 0.2 +
            radar * 0.2 +
            (50 if predicted_class == 1 else 0)
        )

        result = "🚨 DEFECTIVE" if risk_score > 50 else "✅ NON-DEFECTIVE"

        # =======================
        # Result Display
        # =======================
        st.markdown("---")
        st.subheader("📊 Analysis Result")
        st.metric(label="Prediction", value=result)
        st.write(f"**GPS Location:** {lat}, {lon}")
        st.progress(min(risk_score / 100, 1.0))
        st.write(f"**Confidence Score:** {risk_score:.2f} / 100")
