from ultralytics import YOLO
import os

def main():
    # Load the trained model
    model_path = "/home/madan/Machine learning/video_object_detection/runs/detect/train/weights/best.pt"
    model = YOLO(model_path)

    # Define the source image for testing
    # Using an image from the training set to verify the model learned something
    source = "/home/madan/Machine learning/video_object_detection/data/raw/IMG_20251201_203253.jpg"
    
    if not os.path.exists(source):
        print(f"Error: Source file {source} not found.")
        return

    # Run inference
    results = model.predict(source, save=True, conf=0.25)

    print(f"Inference complete. Results saved to {results[0].save_dir}")

if __name__ == '__main__':
    main()
