import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image

# ----------------------------
# Load Track Detection Model
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("railway_model.h5")

model = load_model()

# ----------------------------
# UI Layout
# ----------------------------
st.title("🚄 Railway Safety Monitoring System")
tab1, tab2 = st.tabs(["🧠 Track Fault Detection", "📍 GPS & Collision Prevention"])

# ============================
# TAB 1 - Track Fault Detection
# ============================
with tab1:
    st.header("Detect Defective Railway Tracks")
    uploaded = st.file_uploader("Upload track image", type=["jpg", "png", "jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, caption="Uploaded Track", use_container_width=True)

        img = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img)/255.0, axis=0)

        prediction = model.predict(img_array)[0][0]
        if prediction > 0.5:
            st.error("⚠️ Defective Track Detected")
        else:
            st.success("✅ Track is Properly Aligned")

# ============================
# TAB 2 - Collision Prevention
# ============================
with tab2:
    st.header("Train GPS Tracking & Collision Prevention")

    # Simulated train data
    train_names = [f"Train_{i}" for i in range(1, 11)]
    locations = [
        "Chennai", "Madurai", "Coimbatore", "Trichy", "Salem",
        "Tirunelveli", "Erode", "Thanjavur", "Vellore", "Dindigul"
    ]

    data = []
    for t, loc in zip(train_names, locations):
        km_marker = random.randint(0, 500)
        speed = random.randint(40, 120)
        data.append([t, loc, km_marker, speed])

    df = pd.DataFrame(data, columns=["Train", "Location", "KM_Marker", "Speed"])

    # Detect collision risks
    alerts = []
    safe_distance = 30
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            if abs(df.loc[i,"KM_Marker"] - df.loc[j,"KM_Marker"]) < safe_distance:
                if df.loc[i,"Speed"] > df.loc[j,"Speed"]:
                    alerts.append(f"⚠️ {df.loc[i,'Train']} should SLOW DOWN to avoid collision with {df.loc[j,'Train']}")
                else:
                    alerts.append(f"⚠️ {df.loc[j,'Train']} should SLOW DOWN to avoid collision with {df.loc[i,'Train']}")

    # Scheduling suggestions
    scheduling = []
    for idx, row in df.iterrows():
        if row['Speed'] < 60:
            scheduling.append(f"🕒 {row['Train']} is slow. Schedule next train 15 min later.")
        else:
            scheduling.append(f"✅ {row['Train']} is on time. Schedule next train 5 min later.")

    st.subheader("🚉 Current Train Status")
    st.dataframe(df)

    st.subheader("📢 Collision Alerts")
    if alerts:
        for a in alerts:
            st.error(a)
    else:
        st.success("✅ No collision risks detected")

    st.subheader("📋 Scheduling Suggestions")
    for s in scheduling:
        st.info(s)

    # Graph
    st.subheader("📍 Train Positions")
    fig, ax = plt.subplots()
    ax.scatter(df["KM_Marker"], df["Speed"], c='blue')
    for i, row in df.iterrows():
        ax.text(row["KM_Marker"], row["Speed"]+2, row["Train"], fontsize=8)
    ax.set_xlabel("Track Position (KM)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Train Speed vs Position")
    st.pyplot(fig)
