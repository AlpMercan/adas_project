from ultralytics import YOLO

# Load your model
model = YOLO('models/yolov8n.pt')  # adjust path if your model is elsewhere

# Export to ONNX
model.export(format='onnx', dynamic=True, simplify=True)

# This will create yolov8n.onnx in your current directory
