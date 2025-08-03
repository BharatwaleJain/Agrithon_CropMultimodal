# YOLOv8s Segmented Crop Disease Detection

## Project Directory
```
project/
├── image_dataset/              # Original dataset for processing
├── annotations/                # COCO format annotation from CVAT
│   └── instances_default.json
├── images/default/             # COCO image export from CVAT
├── requirements.txt            # Python dependencies
├── structure.py                # Convert COCO to YOLOv8 format
├── yolov8_disease_dataset/     # Organized dataset (train/val split)
├── augmentation.py             # Apply data augmentation with segmentation masks
├── augmented_disease_dataset/  # Final augmented dataset for training
├── train.py                    # Train YOLOv8s segmentation model
├── runs/                       # Training outputs and predictions
│   ├── detect/ 
│   │   ├── train/              # Training results and model weights
│   │   └── predict/            # Saved annotated images with segmentation masks
├── validation.py               # Validate segmentation model performance
├── test.py                     # Test segmentation model on images
└── web_detector.py             # Streamlit web interface
```

## Commands to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Convert COCO to YOLOv8 segmentation format
python structure.py

# 3. Apply augmentation to training data
python augmentation.py

# 4. Train YOLOv8s segmentation model (100 epochs)
python train.py

# 5. Validate segmentation model metrics
python validation.py

# 6. Test segmentation predictions on validation images
python test.py

# 7. Run Streamlit web interface
streamlit run detector.py
```