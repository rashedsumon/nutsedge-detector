import os
from ultralytics import YOLO

def train_model():
    """
    Call this function locally to train your model.
    It saves the trained weights to 'runs/detect/train/weights/best.pt'
    """
    # Load a pre-trained lightweight YOLOv8 nano model
    model = YOLO("yolov8n.pt") 
    
    # Path to the data.yaml file now located in the root directory
    yaml_path = os.path.join(os.getcwd(), "data.yaml")
    
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Could not find data.yaml at {yaml_path}. Please run data_loader.py first.")
    
    print("Starting training...")
    # Train the model
    model.train(
        data=yaml_path,
        epochs=10,        # Keep epochs low for a quick baseline test
        imgsz=640,        # Standard image size
        device="cpu"      # Forces CPU training (switch to 0 if you have a local GPU)
    )
    print("Training complete! Model saved in the 'runs' directory.")

def predict_image(image_path, model_path="yolov8n.pt"):
    """
    Loads a model and runs an object detection prediction on a single user input image.
    Returns the image with bounding boxes drawn over it.
    """
    # Load the model (defaults to pretrained nano if custom trained weights aren't found)
    if os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        print(f"Custom weights not found at {model_path}. Using generic pretrained YOLOv8 nano.")
        model = YOLO("yolov8n.pt")
        
    # Run inference
    results = model(image_path)
    
    # The first result object contains the plotted bounding boxes image matrix
    annotated_frame = results[0].plot() 
    return annotated_frame

if __name__ == "__main__":
    # Uncomment the line below to run training locally on your machine
    # train_model()
    pass