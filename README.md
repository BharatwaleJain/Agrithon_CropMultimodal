# 🌾 Crop Multimodal – Agrithon Hackathon
A comprehensive agricultural solution developed for Agrithon hackathon, combining computer vision and machine learning for crop disease and insect identification through multiple modalities

---

## System Overview

### **1) Disease Image Model**
YOLOv8s Segmentation for precise crop disease detection with pixel-level segmentation masks

### **2) Insect Image Model** 
YOLOv8s Detection for crop insect identification with bounding box detection

### **3) Disease Text Model**
TabNet Neural Network for symptom-based disease diagnosis through farmer questionnaire responses

### **4) Insect Text Model**
TabNet Neural Network for symptom-based insect identification via visual observation questions

---

## Technology Stack

- **Computer Vision**: YOLOv8s (Ultralytics)
- **Neural Networks**: TabNet (PyTorch-TabNet)
- **Annotation Tool**: CVAT (Offline using Docker Desktop)
- **Web Framework**: Streamlit
- **Data Processing**: OpenCV, Albumentations
- **Machine Learning**: Scikit-learn, Pandas, NumPy

---

# Combined Multimodal Flow

## Crop Image Upload
- Upload single crop image for analysis
- Set confidence thresholds for both disease and insect detection

## Symptom Questions
- Image shown in sidebar during questions  
- Two tabs for Disease Symptoms & Insect Observations  
- Progress indicators track completion status

## Model Analysis
- One button to run all models simultaneously:
  - YOLOv8s Disease Segmentation
  - YOLOv8s Insect Detection  
  - TabNet Disease Text Classification
  - TabNet Insect Text Classification
- Realtime processing with progress spinner

## Final Results
- Crop Disease Present: ✅ / ❌
- Crop Insect Present: ✅ / ❌
- Side by side annotated images  
- Summary of analysis  

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
```

```bash
# Run disease image detection
streamlit run "1) Disease Image Model/detector.py"
```

```bash
# Run insect image detection  
streamlit run "2) Insect Image Model/detector.py"
```

```bash
# Run disease symptom identification
streamlit run "3) Disease Text Model/interface.py"
```

```bash
# Run insect symptom identification
streamlit run "4) Insect Text Model/interface.py"
```

```bash
# Run combined model pipeline analysis
streamlit run combined.py
```

---

## Contributors

- [Naincy Jain](https://www.linkedin.com/in/naincy-jain-38a20a283)
- [Aarjav Jain](https://www.linkedin.com/in/bharatwalejain)