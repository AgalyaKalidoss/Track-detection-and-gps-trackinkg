import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image

# ----------------------------
# Model Path
# ----------------------------
MODEL_PATH = "railway_model_final.tflite"

# ----------------------------
# Load TFLite Model
# ----------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------
# UI Layout
# ----------------------------
st.set_page_config(page_title="🚉 Railway Track Monitoring", layout="wide")
st.title("🚉 Railway Track & Train Safety Monitoring System")

tab1, tab2 = st.tabs(["🧠 Track Fault Detection", "📍 GPS & Sensor Monitoring"])

# ============================
# TAB 1 - Track Fault Detection
# ============================
with tab1:
    st.header("Detect Defective Railway Tracks")
    uploaded = st.file_uploader("Upload track image", type=["jpg", "png", "jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, caption="Uploaded Track", width='stretch')

        # Preprocess
        img_resized = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        st.write("🔍 Raw output:", float(prediction))

        # Threshold: <0.5 -> Defective
        if prediction < 0.5:
            st.error("⚠️ Defective Track Detected")
        else:
            st.success("✅ Track is Properly Aligned")

# ============================
# TAB 2 - GPS & Sensor Monitoring
# ============================
with tab2:
    st.header("Train GPS & Sensor Monitoring")

    # Simulated train and GPS data
    train_names = [f"Train_{i}" for i in range(1, 6)]
    locations = [
        "Chennai", "Madurai", "Coimbatore", "Trichy", "Salem"
    ]

    # Simulated sensor readings (Ultrasonic, Acoustic, Radar, Vibration)
    data = []
    for t, loc in zip(train_names, locations):
        km_marker = random.randint(0, 500)
        speed = random.randint(40, 120)
        ultrasonic = round(random.uniform(0, 5), 2)
        acoustic = round(random.uniform(0, 1), 2)
        radar = round(random.uniform(0, 1), 2)
        vibration = round(random.uniform(0, 1), 2)
        data.append([t, loc, km_marker, speed, ultrasonic, acoustic, radar, vibration])

    df = pd.DataFrame(data, columns=[
        "Train", "Location", "KM_Marker", "Speed",
        "Ultrasonic", "Acoustic", "Radar", "Vibration"
    ])

    st.subheader("🚉 Current Train Status & Sensor Readings")
    st.dataframe(df, width='stretch')

    # Collision detection
    safe_distance = 30
    alerts = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if abs(df.loc[i, "KM_Marker"] - df.loc[j, "KM_Marker"]) < safe_distance:
                if df.loc[i, "Speed"] > df.loc[j, "Speed"]:
                    alerts.append(f"⚠️ {df.loc[i,'Train']} should SLOW DOWN to avoid collision with {df.loc[j,'Train']}")
                else:
                    alerts.append(f"⚠️ {df.loc[j,'Train']} should SLOW DOWN to avoid collision with {df.loc[i,'Train']}")

    st.subheader("📢 Collision Alerts")
    if alerts:
        for a in alerts:
            st.error(a)
    else:
        st.success("✅ No collision risks detected")

    # Scheduling suggestions
    st.subheader("📋 Scheduling Suggestions")
    for idx, row in df.iterrows():
        if row['Speed'] < 60:
            st.info(f"🕒 {row['Train']} is slow. Schedule next train 15 min later.")
        else:
            st.success(f"✅ {row['Train']} is on time. Schedule next train 5 min later.")

    # Visualize train positions vs speed
    st.subheader("📍 Train Positions vs Speed")
    fig, ax = plt.subplots()
    ax.scatter(df["KM_Marker"], df["Speed"], c='blue', s=80)
    for i, row in df.iterrows():
        ax.text(row["KM_Marker"], row["Speed"] + 2, row["Train"], fontsize=9)
    ax.set_xlabel("Track Position (KM)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Train Speed vs Position")
    st.pyplot(fig)
