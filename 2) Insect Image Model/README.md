# YOLOv8s Crop Insect Detection

## Project Directory
```
project/
├── image_dataset/            # Original dataset for processing
├── obj_train_data/           # Original CVAT export
├── requirements.txt          # Python dependencies
├── structure.py              # Organize CVAT export to YOLOv8 format
├── yolov8_insect_dataset/    # Organized dataset (train/val split)
├── augmentation.py           # Apply data augmentation with bounding boxes
├── augmented_insect_dataset/ # Final augmented dataset for training
├── train.py                  # Train YOLOv8s model
├── runs/                     # Training outputs and predictions
│   ├── detect/
│   │   ├── train/            # Training results and model weights
│   │   └── predict/          # Saved annotated images with detections
├── validation.py             # Validate model performance
├── test.py                   # Test model on images
└── web_detector.py           # Streamlit web interface

```

## Commands to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Organize dataset from CVAT export
python structure.py

# 3. Apply augmentation to training data
python augmentation.py

# 4. Train YOLOv8s model (100 epochs)
python train.py

# 5. Validate model metrics
python validation.py

# 6. Test predictions on validation images
python test.py

# 7. Run Streamlit web interface
streamlit run detector.py
```