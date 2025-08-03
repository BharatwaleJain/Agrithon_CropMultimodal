# TabNet Crop Insect Text Classification

## Project Directory
```
project/
├── characteristics.csv                     # 30 insect symptom questions dataset
├── generate.py                             # Generate synthetic training data
├── data.csv                                # Generated synthetic training dataset
├── train.py                                # Train TabNet model
├── crop_insect_tabnet.zip                  # Trained TabNet model weights
├── crop_insect_tabnet_features.pkl         # Feature names for model
├── crop_insect_tabnet_label_encoder.pkl    # Label encoder for insects
├── tabnet_confusion_matrix.png             # Model performance visualization
├── interface.py                            # Streamlit interface
├── sample_insect_images/                   # Reference images
│   ├── aphids.jpg
│   ├── armyworm_brown.jpg
│   ├── armyworm_green.jpg
│   ├── bollworm.jpg
│   ├── cutworm.jpg
│   ├── no_insect.jpg
│   └── whitefly.jpg
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

# 4. Run interface for insect diagnosis
streamlit run interface.py
```