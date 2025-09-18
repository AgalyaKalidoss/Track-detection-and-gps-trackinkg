import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import random

# -------------------------
# Load TFLite model
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
# Page setup
# -------------------------
st.set_page_config(page_title="Railway Safety Monitoring", page_icon="🚦", layout="wide")
st.title("🚦 Railway Safety Monitoring System")
tab1, tab2 = st.tabs(["🧠 Track Fault Detection", "📍 GPS & Collision Prevention"])

# ============================
# TAB 1 - Track Fault Detection
# ============================
with tab1:
    st.header("🛤️ Track Defect Detection")
    uploaded_file = st.file_uploader("📷 Upload a track image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
        st.image(img, caption="Uploaded Image", use_container_width=True)

        img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        threshold = 0.462
        st.write("🔍 **Raw model output:**", round(float(prediction), 4))
        if prediction < threshold:
            st.error("⚠️ **Defective Track Detected** — Please inspect the track.")
        else:
            st.success("✅ **Track is Properly Aligned**")

        st.caption(f"(Using threshold = {threshold})")

# ============================
# TAB 2 - Collision Prevention
# ============================
with tab2:
    st.header("🚉 GPS Based Train Collision Prevention")

    # Simulated train data
    train_names = [f"Train_{i}" for i in range(1, 11)]
    locations = [
        "Chennai", "Madurai", "Coimbatore", "Trichy", "Salem",
        "Tirunelveli", "Erode", "Thanjavur", "Vellore", "Dindigul"
    ]

    data = []
    for t, loc in zip(train_names, locations):
        km_marker = random.randint(0, 500)     # simulated GPS position in km
        speed = random.randint(40, 120)        # speed in km/h
        data.append([t, loc, km_marker, speed])

    df = pd.DataFrame(data, columns=["Train", "Location", "KM_Marker", "Speed"])

    # Detect collision risks
    alerts = []
    safe_distance = 30
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            if abs(df.loc[i, "KM_Marker"] - df.loc[j, "KM_Marker"]) < safe_distance:
                faster = df.loc[i] if df.loc[i, "Speed"] > df.loc[j, "Speed"] else df.loc[j]
                slower = df.loc[j] if df.loc[i, "Speed"] > df.loc[j, "Speed"] else df.loc[i]
                alerts.append(
                    f"⚠️ {faster['Train']} should **SLOW DOWN** to avoid collision with {slower['Train']} (distance < {safe_distance}km)"
                )

    # Scheduling advice
    scheduling = []
    for idx, row in df.iterrows():
        if row['Speed'] < 60:
            scheduling.append(f"🕒 {row['Train']} is slow — schedule next train after **15 min**.")
        else:
            scheduling.append(f"✅ {row['Train']} is on time — schedule next train after **5 min**.")

    # Display results
    st.subheader("📋 Current Train Status")
    st.dataframe(df)

    st.subheader("📢 Collision Alerts")
    if alerts:
        for a in alerts:
            st.error(a)
    else:
        st.success("✅ No collision risks detected")

    st.subheader("📅 Scheduling Suggestions")
    for s in scheduling:
        st.info(s)

    # Graph: speed vs track position
    st.subheader("📍 Train Positions on Track")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df["KM_Marker"], df["Speed"], color='blue')
    for i, row in df.iterrows():
        ax.text(row["KM_Marker"], row["Speed"]+2, row["Train"], fontsize=8)
    ax.set_xlabel("Track Position (KM)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Train Speed vs Position")
    st.pyplot(fig)
