import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Path to your TFLite model
MODEL_PATH = "railway_model_final.tflite"

# Cache the model so it loads only once
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# Preprocess image for model
def preprocess_image(image, input_shape):
    image = image.resize((input_shape[1], input_shape[2]))  # resize to expected input
    img_array = np.array(image).astype(np.float32) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

# Run inference
def predict(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# -------- Streamlit UI --------
st.title("🚉 Railway Track Detection")

interpreter = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']

    input_data = preprocess_image(image, input_shape)
    output = predict(interpreter, input_data)

    # Display prediction result
    st.subheader("Prediction Result")
    st.write("Raw model output:", output)

    # Optional: interpret output as class labels
    classes = ["defective", "non-defective"]
    predicted_class = classes[int(np.argmax(output))]
    confidence = np.max(output)

    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
