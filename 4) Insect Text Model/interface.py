import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib
import streamlit as st
from PIL import Image
import os
class FarmerInsectInterface:
    def __init__(self, model_path='crop_insect_tabnet'):
        self.model = TabNetClassifier()
        self.model.load_model(f'{model_path}.zip')
        self.label_encoder = joblib.load(f'{model_path}_label_encoder.pkl')
        self.feature_columns = joblib.load(f'{model_path}_features.pkl')
        self.questions_df = pd.read_csv('characteristics.csv')
        self.questions = self.questions_df['Question'].tolist()
    def predict_insect(self, responses):
        responses_array = np.array(responses).reshape(1, -1).astype(np.float32)
        prediction = self.model.predict(responses_array)
        probabilities = self.model.predict_proba(responses_array)
        insect_name = self.label_encoder.inverse_transform(prediction)[0]
        confidence = probabilities[0].max()
        all_insects = self.label_encoder.classes_
        insect_probs = {insect: prob for insect, prob in zip(all_insects, probabilities[0])}
        return insect_name, confidence, insect_probs
st.set_page_config(page_title="Crop Insect Detection", layout="wide")
st.title("🐛 Crop Insect Detection System")
st.markdown("Answer the following questions about crop insect symptoms to get insect identification")
try:
    interface = FarmerInsectInterface()
    st.success("✅ Insect TabNet model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()
st.sidebar.header("🦗 Sample Insect Images")
st.sidebar.markdown("*Reference images to help identify insects*")
sample_images_dir = "sample_insect_images"
if os.path.exists(sample_images_dir):
    image_descriptions = {
        'no_insect.jpg': '✅ No Insect - Healthy crop without pests',
        'armyworm_green.jpg': '🟢 Green Armyworm - Green colored larvae',
        'armyworm_brown.jpg': '🟤 Brown Armyworm - Brown colored larvae',
        'cutworm.jpg': '⚫ Cutworm - Cuts stems at soil level',
        'bollworm.jpg': '🔴 Bollworm - Attacks flowers and fruits',
        'aphids.jpg': '🟡 Aphids - Small soft-bodied insects',
        'whitefly.jpg': '⚪ Whitefly - Small white flying insects'
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
    st.sidebar.warning("📁 Sample insect images folder not found")
    st.sidebar.info("""
    **To add sample images:**
    1. Create folder: `sample_insect_images`
    2. Add insect reference images
    3. Restart the application
    """)
st.header("📋 Insect Identification Questions")
st.markdown("Please examine your crop and answer these questions with Yes or No:")
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
    st.header("🎯 Insect Identification Results")
    if st.button("🔍 Identify Insect", type="primary"):
        with st.spinner("Analyzing insect symptoms..."):
            try:
                insect_name, confidence, insect_probs = interface.predict_insect(responses)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Identified Insect", insect_name.replace('_', ' '))
                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")
                with col3:
                    if confidence > 0.7:
                        st.success("High Confidence")
                    elif confidence > 0.5:
                        st.warning("Medium Confidence")
                    else:
                        st.error("Low Confidence")
                st.subheader("📊 All Insect Probabilities")
                sorted_insects = sorted(insect_probs.items(), key=lambda x: x[1], reverse=True)
                for insect, prob in sorted_insects:
                    prob_value = float(prob)
                    st.write(f"**{insect.replace('_', ' ')}**: {prob_value:.1%}")
                    st.progress(prob_value)
                st.subheader("💡 Control Recommendations")
                recommendations = {
                    'No_Insect': "✅ No insects detected! Continue monitoring your crop regularly.",
                    'Armyworm_Green': "🟢 Green Armyworm detected. Apply targeted insecticide, monitor at night.",
                    'Armyworm_Brown': "🟤 Brown Armyworm detected. Use biological control or pesticides.",
                    'Cutworm': "⚫ Cutworm detected. Apply soil treatment, use collar protection.",
                    'Bollworm': "🔴 Bollworm detected. Use pheromone traps, apply recommended pesticides.",
                    'Aphids': "🟡 Aphids detected. Use neem oil, introduce beneficial insects.",
                    'Whitefly': "⚪ Whitefly detected. Use yellow sticky traps, apply systemic insecticides."
                }
                recommendation = recommendations.get(insect_name, "Consult with agricultural extension services.")
                st.info(recommendation)
                if confidence < 0.5:
                    st.warning("⚠️ Low confidence identification. Consider consulting an entomologist.")
                st.subheader("🛡️ General Prevention Tips")
                prevention_tips = """
                - **Regular Monitoring**: Check crops weekly for early detection
                - **Crop Rotation**: Break pest life cycles with different crops
                - **Biological Control**: Encourage natural predators
                - **Proper Sanitation**: Remove crop residues and weeds
                - **Integrated Pest Management**: Combine multiple control methods
                """
                st.markdown(prevention_tips)
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
else:
    missing_count = len([r for r in responses if r is None])
    st.warning(f"⚠️ Please answer all questions. {missing_count} questions remaining.")
st.sidebar.header("ℹ️ About This System")
st.sidebar.info("""
This system uses a TabNet neural network trained on crop insect symptoms to identify pests based on visual observations.

**How to use:**
1. Examine your crop carefully
2. Compare with sample insect images
3. Answer all 30 questions honestly
4. Get instant insect identification
5. Follow the control recommendations

**Note:** This is an identification aid. For severe infestations, consult agricultural experts.
""")