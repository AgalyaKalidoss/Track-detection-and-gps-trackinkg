import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image

# ----------------------------
# Load TFLite Track Detection Model
# ----------------------------
@st.cache_resource
def load_tflite_model(tflite_path="railway_model_final.tflite"):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_tflite_model()

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
    uploaded_file = st.file_uploader("Upload track image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
        st.image(img, caption="Uploaded Track", width='stretch')

        # Preprocess
        img_array = np.expand_dims(np.array(img)/255.0, axis=0).astype(np.float32)

        # Inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        # Handle quantized output if needed
        if output_details[0]['dtype'] != np.float32:
            scale, zero_point = output_details[0]['quantization']
            prediction = scale * (prediction - zero_point)

        # Threshold <50% = Defective
        threshold = 0.5
        if prediction < threshold:
            st.error("⚠️ Defective Track Detected")
        else:
            st.success("✅ Track is Properly Aligned")
        st.write(f"🔍 Prediction Score: {prediction:.3f}")

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
        km_marker = random.randint(0, 500)  # Track position in KM
        speed = random.randint(40, 120)     # Speed in km/h
        data.append([t, loc, km_marker, speed])

    df = pd.DataFrame(data, columns=["Train", "Location", "KM_Marker", "Speed"])

    # Detect collision risks
    alerts = []
    safe_distance = 30  # Minimum safe distance in KM
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

    # Display tables
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
    fig, ax = plt.subplots(figsize=(8,4))
    ax.scatter(df["KM_Marker"], df["Speed"], c='blue')
    for i, row in df.iterrows():
        ax.text(row["KM_Marker"], row["Speed"]+2, row["Train"], fontsize=8)
    ax.set_xlabel("Track Position (KM)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Train Speed vs Position")
    st.pyplot(fig)
