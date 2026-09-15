from ultralytics import YOLO

model = YOLO("best.pt")
print("\n>>> SONARIS model loaded successfully!")
print(">>> Target Classes:", model.names)
