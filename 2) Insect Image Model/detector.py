import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
st.title("Crop Insect Detection")
@st.cache_resource
def load_model():
    try:
        return YOLO('runs/detect/train/weights/best.pt')
    except:
        return None
model = load_model()
if model:
    st.success("Model loaded successfully!")
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    if uploaded_file:
        image = Image.open(uploaded_file)
        if st.button("Detect Insects"):
            with st.spinner("Analyzing..."):
                temp_path = "temp_image.jpg"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                results = model.predict(temp_path, conf=confidence, save=True)
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        num_detections = len(result.boxes)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Original Image")
                            st.image(image, caption="Uploaded Image", use_column_width=True)
                        with col2:
                            st.subheader("Detection Results")
                            if num_detections > 0:
                                annotated_img = result.plot()
                                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                                st.image(annotated_img_rgb, 
                                       caption=f"Detected {num_detections} insects", 
                                       use_column_width=True)
                            else:
                                st.image(image, caption="No detections", use_column_width=True)
                        st.success(f"Detected {num_detections} insects!")
                        if num_detections > 0:
                            confidences = result.boxes.conf.cpu().numpy()
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            with metric_col1:
                                st.metric("Total Insects", num_detections)
                            with metric_col2:
                                st.metric("Avg Confidence", f"{confidences.mean():.2f}")
                            with metric_col3:
                                st.metric("Max Confidence", f"{confidences.max():.2f}")
                            with st.expander("Detailed Detection Results"):
                                boxes = result.boxes.xyxy.cpu().numpy()
                                for i, (conf, box) in enumerate(zip(confidences, boxes)):
                                    x1, y1, x2, y2 = box
                                    st.write(f"**Insect {i+1}:**")
                                    st.write(f"- Confidence: {conf:.2f} ({conf*100:.1f}%)")
                                    st.write(f"- Location: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
                                    st.write("---")
                    else:
                        st.warning("No insects detected. Try lowering confidence threshold.")
                        st.image(image, caption="Original Image (No detections)", use_column_width=True)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            st.subheader("Original Image")
            st.image(image, caption="Click 'Detect Insects' to analyze", use_column_width=True)
else:
    st.error("Model not found! Make sure best.pt exists in runs/detect/train/weights/")