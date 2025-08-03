import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib
import streamlit as st
from PIL import Image
import os
class FarmerDiseaseInterface:
    def __init__(self, model_path='crop_disease_tabnet'):
        self.model = TabNetClassifier()
        self.model.load_model(f'{model_path}.zip')
        self.label_encoder = joblib.load(f'{model_path}_label_encoder.pkl')
        self.feature_columns = joblib.load(f'{model_path}_features.pkl')
        self.questions_df = pd.read_csv('characteristics.csv')
        self.questions = self.questions_df['Question'].tolist()
    def predict_disease(self, responses):
        responses_array = np.array(responses).reshape(1, -1).astype(np.float32)
        prediction = self.model.predict(responses_array)
        probabilities = self.model.predict_proba(responses_array)
        disease_name = self.label_encoder.inverse_transform(prediction)[0]
        confidence = probabilities[0].max()
        all_diseases = self.label_encoder.classes_
        disease_probs = {disease: prob for disease, prob in zip(all_diseases, probabilities[0])}
        return disease_name, confidence, disease_probs
st.set_page_config(page_title="Crop Disease Diagnosis", layout="wide")
st.title("🌱 Crop Disease Diagnosis System")
st.markdown("Answer the following questions about your crop symptoms to get disease prediction")
try:
    interface = FarmerDiseaseInterface()
    st.success("✅ TabNet model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()
st.sidebar.header("📸 Sample Disease Images")
st.sidebar.markdown("*Reference images to help identify symptoms*")
sample_images_dir = "sample_disease_images"
if os.path.exists(sample_images_dir):
    image_descriptions = {
        'healthy_crop.jpg': '🟢 Healthy Crop - No disease symptoms',
        'early_blight.jpg': '🟤 Early Blight - Concentric rings, yellow halo',
        'late_blight.jpg': '🔴 Late Blight - Large brown spots, wilting',
        'bacterial_spot.jpg': '🟡 Bacterial Spot - Small dark spots, yellow halo',
        'septoria_leaf_spot.jpg': '⚫ Septoria Leaf Spot - Small spots, dark centers',
        'target_spot.jpg': '🎯 Target Spot - Circular rings, target pattern',
        'leaf_mold.jpg': '🟫 Leaf Mold - Yellow leaves, black growth'
    }
    for img_file in sorted(os.listdir(sample_images_dir)):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(sample_images_dir, img_file)
            try:
                image = Image.open(img_path)
                description = image_descriptions.get(img_file, img_file)
                with st.sidebar.expander(description, expanded=False):
                    st.image(image, width=250)
            except Exception as e:
                st.sidebar.error(f"Error loading {img_file}")
else:
    st.sidebar.warning("📁 Sample images folder not found")
    st.sidebar.info("""
    **To add sample images:**
    1. Create folder: `sample_disease_images`
    2. Add disease reference images
    3. Restart the application
    """)
st.header("📋 Disease Symptom Questions")
st.markdown("Please look at your crop and answer these questions with Yes or No:")
responses = []
col1, col2 = st.columns(2)
for i, question in enumerate(interface.questions):
    current_col = col1 if i % 2 == 0 else col2
    with current_col:
        with st.container():
            st.markdown(f"**Question {i+1}:**")
            st.markdown(f"*{question}*")
            response = st.radio(
                f"Answer for Question {i+1}:",
                options=["Select", "Yes", "No"],
                key=f"q_{i}",
                horizontal=True
            )
            if response == "Yes":
                responses.append(1)
            elif response == "No":
                responses.append(0)
            else:
                responses.append(None)
            st.markdown("---")
if len([r for r in responses if r is not None]) == len(interface.questions):
    st.header("🎯 Disease Prediction Results")
    if st.button("🔍 Diagnose Disease", type="primary"):
        with st.spinner("Analyzing symptoms..."):
            try:
                disease_name, confidence, disease_probs = interface.predict_disease(responses)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Disease", disease_name)
                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")
                with col3:
                    if confidence > 0.7:
                        st.success("High Confidence")
                    elif confidence > 0.5:
                        st.warning("Medium Confidence")
                    else:
                        st.error("Low Confidence")
                st.subheader("📊 All Disease Probabilities")
                sorted_diseases = sorted(disease_probs.items(), key=lambda x: x[1], reverse=True)
                for disease, prob in sorted_diseases:
                    prob_value = float(prob)
                    st.write(f"**{disease.replace('_', ' ')}**: {prob_value:.1%}")
                    st.progress(prob_value)
                st.subheader("💡 Recommendations")
                recommendations = {
                    'Healthy': "✅ Your crop appears healthy! Continue good agricultural practices.",
                    'Early_Blight': "🔶 Apply fungicide treatment and improve air circulation. Remove affected leaves.",
                    'Late_Blight': "🔴 Immediate fungicide treatment needed. Avoid overhead watering.",
                    'Bacterial_Spot': "🟡 Use copper-based bactericides. Improve field sanitation.",
                    'Septoria_Leaf_Spot': "🟠 Apply fungicide and remove lower leaves. Improve air circulation.",
                    'Target_Spot': "🟤 Fungicide treatment recommended. Practice crop rotation.",
                    'Leaf_Mold': "🟢 Improve ventilation and reduce humidity. Apply fungicide if severe."
                }
                recommendation = recommendations.get(disease_name, "Consult with agricultural extension services.")
                st.info(recommendation)
                if confidence < 0.5:
                    st.warning("⚠️ Low confidence prediction. Consider consulting an agricultural expert for accurate diagnosis.")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
else:
    missing_count = len([r for r in responses if r is None])
    st.warning(f"⚠️ Please answer all questions. {missing_count} questions remaining.")
st.sidebar.header("ℹ️ About This System")
st.sidebar.info("""
This system uses a TabNet neural network trained on crop disease symptoms to predict diseases based on visual observations.

**How to use:**
1. Observe your crop carefully
2. Answer all 30 questions honestly
3. Get instant disease prediction
4. Follow the recommendations

**Note:** This is a diagnostic aid. For severe infections, consult agricultural experts.
""")