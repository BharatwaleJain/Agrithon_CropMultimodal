import streamlit as st
import pandas as pd
import numpy as np
from ultralytics import YOLO
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib
from PIL import Image
import cv2
import os
class CombinedCropAnalyzer:
    def __init__(self):
        self.load_models()
        self.load_questions()
    def load_models(self):
        try:
            self.disease_img_model = YOLO('1) Disease Image Model/runs/segment/train/weights/best.pt')
            st.sidebar.success("✅ Disease Image Model Loaded")
        except:
            self.disease_img_model = None
            st.sidebar.error("❌ Disease Image Model Not Found")
        try:
            self.insect_img_model = YOLO('2) Insect Image Model/runs/detect/train/weights/best.pt')
            st.sidebar.success("✅ Insect Image Model Loaded")
        except:
            self.insect_img_model = None
            st.sidebar.error("❌ Insect Image Model Not Found")
        try:
            self.disease_text_model = TabNetClassifier()
            self.disease_text_model.load_model('3) Disease Text Model/crop_disease_tabnet.zip')
            self.disease_label_encoder = joblib.load('3) Disease Text Model/crop_disease_tabnet_label_encoder.pkl')
            st.sidebar.success("✅ Disease Text Model Loaded")
        except:
            self.disease_text_model = None
            st.sidebar.error("❌ Disease Text Model Not Found")
        try:
            self.insect_text_model = TabNetClassifier()
            self.insect_text_model.load_model('4) Insect Text Model/crop_insect_tabnet.zip')
            self.insect_label_encoder = joblib.load('4) Insect Text Model/crop_insect_tabnet_label_encoder.pkl')
            st.sidebar.success("✅ Insect Text Model Loaded")
        except:
            self.insect_text_model = None
            st.sidebar.error("❌ Insect Text Model Not Found")
    def load_questions(self):
        try:
            self.disease_questions = pd.read_csv('3) Disease Text Model/characteristics.csv')['Question'].tolist()
        except:
            self.disease_questions = [f"Disease symptom question {i+1}" for i in range(30)]
        try:
            self.insect_questions = pd.read_csv('4) Insect Text Model/characteristics.csv')['Question'].tolist()
        except:
            self.insect_questions = [f"Insect observation question {i+1}" for i in range(30)]
st.set_page_config(page_title="Combined Crop Analysis", layout="wide")
st.title("🌾 Crop Multimodal – Agrithon Hackathon")
analyzer = CombinedCropAnalyzer()
st.header("📸 Step 1: Upload Crop Image")
uploaded_file = st.file_uploader("Choose an image for complete analysis", type=['jpg', 'jpeg', 'png'])
if not uploaded_file:
    st.info("👆 Please upload a crop image to start the analysis flow")
    st.markdown(
        """
        <hr style="margin-top: 2em; margin-bottom: 0.5em;">
        <div style="text-align: center;">
            Made with &hearts; by Naincy and Aarjav Jain
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()
image = Image.open(uploaded_file)
st.image(image, use_container_width=True)
st.success("✅ Image uploaded successfully! Now proceed with symptom questions.")
col1, col2 = st.columns(2)
with col1:
    disease_confidence = st.slider("Disease Detection Confidence", 0.1, 1.0, 0.5)
with col2:
    insect_confidence = st.slider("Insect Detection Confidence", 0.1, 1.0, 0.5)
st.header("📋 Step 2: Answer Questions")
disease_tab, insect_tab = st.tabs(["🦠 Disease Symptoms", "🐛 Insect Observations"])
disease_responses = []
insect_responses = []
with st.sidebar:
    st.header("📸 Uploaded Image")
    st.image(image, use_container_width=True)
with disease_tab:
    st.subheader("Disease Symptom Questions")
    with st.sidebar:
        st.header("🦠 Disease Reference Images")
        sample_disease_dir = "3) Disease Text Model/sample_disease_images"
        if os.path.exists(sample_disease_dir):
            disease_image_descriptions = {
                'healthy_crop.jpg': '🟢 Healthy Crop - No disease symptoms',
                'early_blight.jpg': '🟤 Early Blight - Concentric rings, yellow halo',
                'late_blight.jpg': '🔴 Late Blight - Large brown spots, wilting',
                'bacterial_spot.jpg': '🟡 Bacterial Spot - Small dark spots, yellow halo',
                'septoria_leaf_spot.jpg': '⚫ Septoria Leaf Spot - Small spots, dark centers',
                'target_spot.jpg': '🎯 Target Spot - Circular rings, target pattern',
                'leaf_mold.jpg': '🟫 Leaf Mold - Yellow leaves, black growth'
            }
            for img_file in sorted(os.listdir(sample_disease_dir)):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(sample_disease_dir, img_file)
                    try:
                        sample_image = Image.open(img_path)
                        description = disease_image_descriptions.get(img_file, img_file)
                        with st.expander(description, expanded=False):
                            st.image(sample_image, width=200)
                    except Exception as e:
                        st.error(f"Error loading {img_file}")
        else:
            st.warning("Disease sample images not found")
    col1, col2 = st.columns(2)
    for i, question in enumerate(analyzer.disease_questions):
        current_col = col1 if i % 2 == 0 else col2
        with current_col:
            st.markdown(f"**Q{i+1}:** {question}")
            response = st.radio(
                f"Answer {i+1}:",
                ["Select", "Yes", "No"],
                key=f"disease_q_{i}",
                horizontal=True
            )
            if response == "Yes":
                disease_responses.append(1)
            elif response == "No":
                disease_responses.append(0)
            else:
                disease_responses.append(None)
with insect_tab:
    st.subheader("Insect Observation Questions")
    with st.sidebar:
        st.header("🐛 Insect Reference Images")
        sample_insect_dir = "4) Insect Text Model/sample_insect_images"
        if os.path.exists(sample_insect_dir):
            insect_image_descriptions = {
                'no_insect.jpg': '✅ No Insect - Healthy crop without pests',
                'armyworm_green.jpg': '🟢 Green Armyworm - Green colored larvae',
                'armyworm_brown.jpg': '🟤 Brown Armyworm - Brown colored larvae',
                'cutworm.jpg': '⚫ Cutworm - Cuts stems at soil level',
                'bollworm.jpg': '🔴 Bollworm - Attacks flowers and fruits',
                'aphids.jpg': '🟡 Aphids - Small soft-bodied insects',
                'whitefly.jpg': '⚪ Whitefly - Small white flying insects'
            }
            for img_file in sorted(os.listdir(sample_insect_dir)):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(sample_insect_dir, img_file)
                    try:
                        sample_image = Image.open(img_path)
                        description = insect_image_descriptions.get(img_file, img_file)
                        with st.expander(description, expanded=False):
                            st.image(sample_image, width=200)
                    except Exception as e:
                        st.error(f"Error loading {img_file}")
        else:
            st.warning("Insect sample images not found")
    col1, col2 = st.columns(2)
    for i, question in enumerate(analyzer.insect_questions):
        current_col = col1 if i % 2 == 0 else col2
        with current_col:
            st.markdown(f"**Q{i+1}:** {question}")
            response = st.radio(
                f"Answer {i+1}:",
                ["Select", "Yes", "No"],
                key=f"insect_q_{i}",
                horizontal=True
            )
            if response == "Yes":
                insect_responses.append(1)
            elif response == "No":
                insect_responses.append(0)
            else:
                insect_responses.append(None)
st.header("🔬 Step 3: Run Complete Analysis")
disease_complete = all(r is not None for r in disease_responses)
insect_complete = all(r is not None for r in insect_responses)
col1, col2 = st.columns(2)
with col1:
    disease_progress = len([r for r in disease_responses if r is not None]) / len(disease_responses) if disease_responses else 0
    st.progress(disease_progress)
    st.write(f"Disease Questions: {disease_progress:.0%} Complete")
with col2:
    insect_progress = len([r for r in insect_responses if r is not None]) / len(insect_responses) if insect_responses else 0
    st.progress(insect_progress)
    st.write(f"Insect Questions: {insect_progress:.0%} Complete")
if st.button("🚀 Run Complete Analysis", type="primary", disabled=not (disease_complete and insect_complete)):
    st.header("🎯 Step 4: Analysis Results")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Analyzed Crop Image", use_container_width=True)
    with st.spinner("Running multimodal analysis..."):
        temp_path = "temp_analysis.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        disease_img_result = {'disease_detected': False}
        insect_img_result = {'insect_detected': False}
        if analyzer.disease_img_model:
            results = analyzer.disease_img_model.predict(temp_path, conf=disease_confidence)
            for result in results:
                if hasattr(result, 'masks') and result.masks is not None:
                    num_detections = len(result.masks)
                    confidences = result.boxes.conf.cpu().numpy()
                    disease_img_result = {
                        'disease_detected': num_detections > 0,
                        'num_detections': num_detections,
                        'avg_confidence': float(confidences.mean()) if len(confidences) > 0 else 0,
                        'annotated_img': result.plot()
                    }
        if analyzer.insect_img_model:
            results = analyzer.insect_img_model.predict(temp_path, conf=insect_confidence)
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    num_detections = len(result.boxes)
                    confidences = result.boxes.conf.cpu().numpy()
                    insect_img_result = {
                        'insect_detected': num_detections > 0,
                        'num_detections': num_detections,
                        'avg_confidence': float(confidences.mean()) if len(confidences) > 0 else 0,
                        'annotated_img': result.plot()
                    }
        disease_text_result = {'disease_present': False}
        insect_text_result = {'insect_present': False}
        if analyzer.disease_text_model and disease_complete:
            responses_array = np.array(disease_responses).reshape(1, -1).astype(np.float32)
            prediction = analyzer.disease_text_model.predict(responses_array)
            probabilities = analyzer.disease_text_model.predict_proba(responses_array)
            disease_name = analyzer.disease_label_encoder.inverse_transform(prediction)[0]
            confidence = float(probabilities[0].max())
            disease_text_result = {
                'disease_present': disease_name.lower() != 'healthy',
                'disease_name': disease_name,
                'confidence': confidence
            }
        if analyzer.insect_text_model and insect_complete:
            responses_array = np.array(insect_responses).reshape(1, -1).astype(np.float32)
            prediction = analyzer.insect_text_model.predict(responses_array)
            probabilities = analyzer.insect_text_model.predict_proba(responses_array)
            insect_name = analyzer.insect_label_encoder.inverse_transform(prediction)[0]
            confidence = float(probabilities[0].max())
            insect_text_result = {
                'insect_present': insect_name.lower() not in ['no_insect', 'healthy'],
                'insect_name': insect_name,
                'confidence': confidence
            }
        st.success("✅ Analysis Complete!")
        final_disease_present = disease_img_result.get('disease_detected', False) or disease_text_result.get('disease_present', False)
        final_insect_present = insect_img_result.get('insect_detected', False) or insect_text_result.get('insect_present', False)
        st.subheader("🎯 **FINAL OUTPUT**")
        col1, col2 = st.columns(2)
        with col1:
            disease_status = "PRESENT" if final_disease_present else "NOT PRESENT"
            color = "🔴" if final_disease_present else "🟢"
            st.metric(f"{color} Crop Disease", disease_status)
        with col2:
            insect_status = "PRESENT" if final_insect_present else "NOT PRESENT"
            color = "🔴" if final_insect_present else "🟢"
            st.metric(f"{color} Crop Insect", insect_status)
        st.subheader("📊 Visual Analysis Results")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Disease Detection:**")
            if 'annotated_img' in disease_img_result:
                annotated_disease = cv2.cvtColor(disease_img_result['annotated_img'], cv2.COLOR_BGR2RGB)
                st.image(annotated_disease, caption="Disease Segmentation", use_container_width=True)
            else:
                st.image(image, caption="No Disease Detected", use_container_width=True)
        with col2:
            st.write("**Insect Detection:**")
            if 'annotated_img' in insect_img_result:
                annotated_insect = cv2.cvtColor(insect_img_result['annotated_img'], cv2.COLOR_BGR2RGB)
                st.image(annotated_insect, caption="Insect Detection", use_container_width=True)
            else:
                st.image(image, caption="No Insects Detected", use_container_width=True)
        st.subheader("Detailed Analysis")
        with st.expander("View Complete Results"):
            st.json({
                'Final_Output': {
                    'Crop_Disease_Present': final_disease_present,
                    'Crop_Insect_Present': final_insect_present
                },
                'Image_Analysis': {
                    'Disease_Detected': disease_img_result.get('disease_detected', False),
                    'Disease_Count': disease_img_result.get('num_detections', 0),
                    'Insect_Detected': insect_img_result.get('insect_detected', False),
                    'Insect_Count': insect_img_result.get('num_detections', 0)
                },
                'Text_Analysis': {
                    'Disease_Diagnosis': disease_text_result.get('disease_name', 'Unknown'),
                    'Disease_Confidence': disease_text_result.get('confidence', 0),
                    'Insect_Identification': insect_text_result.get('insect_name', 'Unknown'),
                    'Insect_Confidence': insect_text_result.get('confidence', 0)
                }
            })
        if os.path.exists(temp_path):
            os.remove(temp_path)
else:
    remaining = []
    if not disease_complete:
        remaining.append("Disease symptoms")
    if not insect_complete:
        remaining.append("Insect observations")
    st.warning(f"⚠️ Please complete all questions for: {', '.join(remaining)}")
st.markdown(
    """
    <hr style="margin-top: 2em; margin-bottom: 0.5em;">
    <div style="text-align: center;">
        Made with &hearts; by Naincy and Aarjav Jain
    </div>
    """,
    unsafe_allow_html=True
)