from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    # We use the absolute path to the data.yaml file
    results = model.train(data="/home/madan/Machine learning/video_object_detection/data/Annoted/data.yaml", epochs=50, imgsz=640)

if __name__ == '__main__':
    main()
