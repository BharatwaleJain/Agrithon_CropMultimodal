from ultralytics import YOLO
model = YOLO('yolov8s-seg.pt')
results = model.train(
    data='augmented_disease_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    lr0=0.01,
    device='cpu',
    patience=20,
    task='segment'
)
print("Training completed!")
print(f"Best model saved to: runs/segment/train/weights/best.pt")