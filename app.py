import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import geocoder
import io

# ------------------------
# Load TFLite Model
# ------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="railway_model_final.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------------
# Helper Functions
# ------------------------
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(img):
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def get_gps_location():
    try:
        g = geocoder.ip('me')
        if g.ok:
            return g.latlng
        else:
            return None
    except:
        return None

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Railway Track Fault Detector", page_icon="🚆", layout="wide")

st.title("🚆 Railway Track Fault Detection with GPS")
st.write("Upload a railway track image to check if it is **Defective** or **Non-Defective** along with GPS location.")

# Upload Section
uploaded_file = st.file_uploader("📷 Upload a Railway Track Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict Button
    if st.button("🔍 Detect Fault"):
        # Preprocess and Predict
        input_img = preprocess_image(image)
        result = predict(input_img)
        confidence = float(np.max(result))
        predicted_class = np.argmax(result)

        label = "Defective" if predicted_class == 1 else "Non-Defective"
        color = "red" if predicted_class == 1 else "green"

        st.markdown(f"### ✅ Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
        st.progress(confidence)

        # Get GPS
        gps = get_gps_location()
        if gps:
            st.success(f"📍 GPS Location: Latitude {gps[0]}, Longitude {gps[1]}")
        else:
            st.warning("⚠️ GPS not found automatically. Please enter manually below:")

        lat = st.text_input("Manual Latitude (if GPS failed)")
        lon = st.text_input("Manual Longitude (if GPS failed)")
        if lat and lon:
            st.info(f"📍 Manual Location: Latitude {lat}, Longitude {lon}")
