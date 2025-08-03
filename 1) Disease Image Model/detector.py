import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
st.title("Crop Disease Spot Segmentation")
@st.cache_resource
def load_model():
    try:
        return YOLO('runs/segment/train/weights/best.pt')
    except:
        return None
model = load_model()
if model:
    st.success("YOLOv8s Segmentation model loaded successfully!")
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    if uploaded_file:
        image = Image.open(uploaded_file)
        if st.button("Detect Disease Spots"):
            with st.spinner("Analyzing for disease spots..."):
                temp_path = "temp_disease_image.jpg"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                results = model.predict(temp_path, conf=confidence, save=True)
                for result in results:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Image")
                        st.image(image, caption="Uploaded Image", use_column_width=True)
                    with col2:
                        st.subheader("Disease Spot Detection")
                        if hasattr(result, 'masks') and result.masks is not None:
                            num_detections = len(result.masks)
                            annotated_img = result.plot()
                            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                            st.image(annotated_img_rgb, caption=f"Detected {num_detections} disease spots", use_column_width=True)
                        else:
                            st.image(image, caption="No disease spots detected", use_column_width=True)
                    if hasattr(result, 'masks') and result.masks is not None:
                        num_detections = len(result.masks)
                        st.success(f"Detected {num_detections} disease spots!")
                        if num_detections > 0:
                            confidences = result.boxes.conf.cpu().numpy()
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            with metric_col1:
                                st.metric("Disease Spots", num_detections)
                            with metric_col2:
                                st.metric("Avg Confidence", f"{confidences.mean():.2f}")
                            with metric_col3:
                                st.metric("Max Confidence", f"{confidences.max():.2f}")
                    else:
                        st.warning("No disease spots detected. Try lowering confidence threshold.")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            st.subheader("Original Image")
            st.image(image, caption="Click 'Detect Disease Spots' to analyze", use_column_width=True)
else:
    st.error("Segmentation model not found!")