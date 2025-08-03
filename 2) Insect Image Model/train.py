from ultralytics import YOLO
model = YOLO('yolov8s.pt')
results = model.train(
    data='augmented_insect_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    lr0=0.01,
    device='cpu',
    patience=20,
    project='runs/detect',
    name='train'
)
print("Training completed!")
print(f"Best model saved to: runs/detect/train/weights/best.pt")