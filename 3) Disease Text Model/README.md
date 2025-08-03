# TabNet Crop Disease Text Classification

## Project Directory
```
project/
├── characteristics.csv                     # 30 disease symptom questions dataset
├── generate.py                             # Generate synthetic training data
├── data.csv                                # Generated synthetic training dataset
├── train.py                                # Train TabNet model
├── crop_disease_tabnet.zip                 # Trained TabNet model weights
├── crop_disease_tabnet_features.pkl        # Feature names for model
├── crop_disease_tabnet_label_encoder.pkl   # Label encoder for diseases
├── tabnet_confusion_matrix.png             # Model performance visualization
├── interface.py                            # Streamlit interface
├── sample_disease_images/                  # Reference images
│   ├── bacterial_spot.jpg
│   ├── early_blight.jpg
│   ├── healthy_crop.jpg
│   ├── late_blight.jpg
│   ├── leaf_mold.jpg
│   └── septoria_leaf_spot.jpg
└── requirements.txt                        # Python dependencies
```

## Commands to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic training data from 30 questions
python generate.py

# 3. Train TabNet model on synthetic data
python train.py

# 4. Run interface for disease diagnosis
streamlit run interface.py
```