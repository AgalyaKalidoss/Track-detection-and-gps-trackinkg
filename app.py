import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image

# 🎨 Page Settings
st.set_page_config(page_title="Railway Safety System", page_icon="🚄", layout="wide")

# ----------------------------
# Load Track Detection Model
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("railway_model.h5")

model = load_model()

# ============================
# Title & Tabs
# ============================
st.title("🚄 Indian Railway Safety Monitoring System")
st.markdown("Ensuring **safe journeys** through AI-powered track detection and real-time train monitoring ⚡")

tab1, tab2 = st.tabs(["🧠 Track Fault Detection", "📍 GPS & Collision Prevention"])

# ============================
# TAB 1 - Track Fault Detection
# ============================
with tab1:
    st.header("🧠 Detect Defective Railway Tracks")

    uploaded = st.file_uploader("📸 Upload track image", type=["jpg", "png", "jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, caption="Uploaded Track Image", use_container_width=True)

        # Preprocess
        img = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # Predict
        with st.spinner("Analyzing track..."):
            prediction = model.predict(img_array)[0][0]

        st.divider()
        if prediction > 0.5:
            st.error("⚠️ **Defective Track Detected!** Immediate inspection required.")
        else:
            st.success("✅ **Track is Properly Aligned and Safe.**")

# ============================
# TAB 2 - Collision Prevention
# ============================
with tab2:
    st.header("📍 Real-Time Train GPS Tracking & Collision Prevention")

    # Generate sample train data
    train_names = [f"🚆 Train_{i}" for i in range(1, 11)]
    locations = [
        "Chennai", "Madurai", "Coimbatore", "Trichy", "Salem",
        "Tirunelveli", "Erode", "Thanjavur", "Vellore", "Dindigul"
    ]

    data = []
    for t, loc in zip(train_names, locations):
        km_marker = random.randint(0, 500)
        speed = random.randint(40, 120)
        data.append([t, loc, km_marker, speed])

    df = pd.DataFrame(data, columns=["Train", "Location", "KM_Marker", "Speed (km/h)"])

    # Detect collision risks
    alerts = []
    safe_distance = 30
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if abs(df.loc[i, "KM_Marker"] - df.loc[j, "KM_Marker"]) < safe_distance:
                if df.loc[i, "Speed (km/h)"] > df.loc[j, "Speed (km/h)"]:
                    alerts.append(f"⚠️ {df.loc[i,'Train']} should **SLOW DOWN** to avoid collision with {df.loc[j,'Train']}")
                else:
                    alerts.append(f"⚠️ {df.loc[j,'Train']} should **SLOW DOWN** to avoid collision with {df.loc[i,'Train']}")

    # Scheduling suggestions
    scheduling = []
    for idx, row in df.iterrows():
        if row['Speed (km/h)'] < 60:
            scheduling.append(f"🕒 {row['Train']} is **slow** — Schedule next train **15 min later**.")
        else:
            scheduling.append(f"✅ {row['Train']} is **on time** — Schedule next train **5 min later**.")

    # Display Table
    st.subheader("🚉 Current Train Status")
    st.dataframe(df.style.background_gradient(cmap="Blues"))

    # Collision Alerts
    st.subheader("📢 Collision Risk Alerts")
    if alerts:
        for a in alerts:
            st.error(a)
    else:
        st.success("✅ No collision risks detected — All trains are safely spaced.")

    # Scheduling
    st.subheader("📋 Scheduling Suggestions")
    for s in scheduling:
        st.info(s)

    # Graph
    st.subheader("📊 Train Speed vs Track Position")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df["KM_Marker"], df["Speed (km/h)"], c='royalblue', s=60)
    for i, row in df.iterrows():
        ax.text(row["KM_Marker"], row["Speed (km/h)"] + 2, row["Train"], fontsize=8, ha='center')
    ax.set_xlabel("Track Position (KM)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("📍 Train Positions on Track")
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
