import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

import folium
from streamlit_folium import st_folium

# -------------------------
# Load TFLite Model
# -------------------------
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------
# Page Setup
# -------------------------
st.set_page_config(page_title="Railway Safety System", page_icon="🚉", layout="wide")
st.title("🚉 Railway Safety Monitoring System")

tab1, tab2 = st.tabs(["🧠 Track Fault Detection", "📍 GPS Train Location"])

# =========================
# Tab 1: Track Fault Detection
# =========================
with tab1:
    st.header("🛤️ Detect Defective Railway Tracks")
    uploaded_file = st.file_uploader("Upload a track image", type=["jpg","jpeg","png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Track", use_container_width=True)

        # Preprocess
        img = image.resize((224,224))
        img_array = np.expand_dims(np.array(img), axis=0).astype(np.float32)
        # Apply MobileNetV2 preprocessing
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img_array = preprocess_input(img_array)

        # Inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        # Sigmoid output
        prob = float(output[0])
        threshold = 0.5  # 50% threshold
        if prob < threshold:
            st.error(f"🚨 Defective Track Detected\nConfidence: {prob:.2%}")
        else:
            st.success(f"✅ Track is Properly Aligned\nConfidence: {prob:.2%}")

# =========================
# Tab 2: GPS / Collision Prevention
# =========================
with tab2:
    st.header("📍 GPS Train Location & Collision Simulation")

    # Simulated train data
    train_names = [f"Train_{i}" for i in range(1,11)]
    locations = [
        "Chennai","Madurai","Coimbatore","Trichy","Salem",
        "Tirunelveli","Erode","Thanjavur","Vellore","Dindigul"
    ]

    data = []
    for t, loc in zip(train_names, locations):
        km_marker = np.random.randint(0,500)
        speed = np.random.randint(40,120)
        lat = np.random.uniform(8.0,13.0)
        lon = np.random.uniform(76.0,80.0)
        data.append([t, loc, km_marker, speed, lat, lon])

    df = pd.DataFrame(data, columns=["Train","Location","KM_Marker","Speed","Latitude","Longitude"])
    st.subheader("🚉 Current Train Status")
    st.dataframe(df)

    # Collision Alerts
    alerts = []
    safe_distance = 30
    for i in range(len(df)):
        for j in range(i+1,len(df)):
            if abs(df.loc[i,"KM_Marker"] - df.loc[j,"KM_Marker"]) < safe_distance:
                faster = df.loc[i] if df.loc[i,"Speed"] > df.loc[j,"Speed"] else df.loc[j]
                slower = df.loc[j] if df.loc[i,"Speed"] > df.loc[j,"Speed"] else df.loc[i]
                alerts.append(f"⚠️ {faster['Train']} should SLOW DOWN to avoid collision with {slower['Train']}")

    st.subheader("📢 Collision Alerts")
    if alerts:
        for a in alerts:
            st.error(a)
    else:
        st.success("✅ No collision risks detected")

    # Scheduling
    st.subheader("📋 Scheduling Suggestions")
    for idx, row in df.iterrows():
        if row["Speed"] < 60:
            st.info(f"🕒 {row['Train']} is slow. Schedule next train 15 min later.")
        else:
            st.info(f"✅ {row['Train']} is on time. Schedule next train 5 min later.")

    # Map Display
    st.subheader("🗺️ Train Positions on Map")
    m = folium.Map(location=[11.0,78.0], zoom_start=6)
    for idx,row in df.iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=f"{row['Train']} ({row['Speed']} km/h)",
            tooltip=row["Train"],
            icon=folium.Icon(color='blue' if row["Speed"]>=60 else 'red')
        ).add_to(m)
    st_folium(m, width=700, height=500)
