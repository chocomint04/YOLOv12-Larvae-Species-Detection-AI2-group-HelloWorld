import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np


# Streamlit page setup
st.set_page_config(page_title="🦟 Larvae Detection Demo", layout="wide")
st.title("🦟 Larvae Detection using YOLOv12")

# Loading the model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # make sure best.pt is in the same folder
    return model

model = load_model()
st.success("Model loaded successfully!")
# File uploader
uploaded_file = st.file_uploader("Upload an image for detection", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run inference
    with st.spinner("Detecting larvae species..."):
        results = model.predict(image, imgsz=800, conf=0.5)
        result_img = results[0].plot()  # draws boxes and masks

    st.image(result_img, caption="Detection Result", use_column_width=True)

    # For detections detail display
    st.subheader("Detection Details")
    if len(results[0].boxes) == 0:
        st.info("No larvae detected.")
    else:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"**{model.names[cls]}** — Confidence: {conf:.2f}")

