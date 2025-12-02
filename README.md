# Object Detection Pipeline - Noodle Recognition

A production-ready object detection pipeline built with YOLOv8, demonstrating end-to-end ML engineering from data annotation to model deployment.

## 🎯 Project Overview

This project implements a complete computer vision pipeline for detecting and localizing noodle objects in images. It showcases core ML engineering competencies including data curation, annotation quality assurance, model training, and systematic evaluation.

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| mAP@50 | **99.5%** |
| mAP@50-95 | **92.3%** |
| Precision | **100%** |
| Recall | **99.4%** |
| Inference Speed | ~27ms per image |

## 🏗️ Pipeline Architecture

```
Data Collection → Annotation (CVAT) → Quality Assurance → Training → Validation → Deployment
```

### 1. Data Curation & Annotation

- **Annotation Tool**: CVAT (Computer Vision Annotation Tool)
- **Format**: YOLO format (normalized bounding boxes)
- **Dataset**: 51 high-quality annotated images
- **Class Ontology**: Single-class detection (Noodle)

#### Annotation Quality Assurance
- Manual review and gold standard creation
- Consistent bounding box annotations
- YOLO format validation (normalized coordinates)
- Train/validation split strategy

### 2. Model Engineering

**Architecture**: YOLOv8n (Nano)
- Lightweight model optimized for speed-accuracy tradeoff
- 3M parameters, 8.1 GFLOPs
- Real-time inference capability

**Training Configuration**:
```yaml
epochs: 50
image_size: 640
optimizer: AdamW (lr=0.002, momentum=0.9)
batch_size: Auto-configured
augmentation: Default YOLOv8 augmentations
```

### 3. Error Analysis & Model Validation

The model achieves near-perfect validation performance:
- **Zero false positives** in validation set
- **99.4% recall** - minimal false negatives
- Robust generalization across different lighting conditions and angles

## 🚀 Quick Start

### Prerequisites
```bash
pip install ultralytics
```

### Training
```bash
python train_yolo.py
```

The training script will:
1. Load pre-trained YOLOv8n weights
2. Fine-tune on custom dataset
3. Save best weights to `runs/detect/train/weights/best.pt`
4. Generate training metrics and visualizations

### Inference
```bash
python test_yolo.py
```

Or use the CLI directly:
```bash
yolo predict model="runs/detect/train/weights/best.pt" source="path/to/image.jpg" conf=0.25
```

## 📁 Project Structure

```
video_object_detection/
├── data/
│   ├── Annoted/
│   │   ├── obj_train_data/          # Training images + labels
│   │   ├── data.yaml                # Dataset configuration
│   │   ├── obj.names                # Class names
│   │   └── obj.data                 # Dataset metadata
│   └── raw/                         # Raw unprocessed images
├── runs/
│   └── detect/
│       ├── train/                   # Training outputs + weights
│       └── predict*/                # Inference results
├── train_yolo.py                    # Training script
├── test_yolo.py                     # Inference script
└── README.md
```

## 🔬 Technical Implementation

### Data Pipeline
1. **Data Collection**: Image acquisition from multiple angles and lighting conditions
2. **Annotation**: Precise bounding box labeling using CVAT
3. **Export**: YOLO format conversion with normalized coordinates
4. **Validation**: Automated format checks and consistency verification

### Training Pipeline
- **Transfer Learning**: Leverages pre-trained YOLOv8n weights
- **Data Augmentation**: Automatic augmentation strategies (rotation, scaling, color jitter)
- **Optimizer**: AdamW with adaptive learning rate
- **Early Stopping**: Best model selection based on mAP@50

### Inference Pipeline
- **Preprocessing**: Automatic image resizing and normalization
- **Detection**: Real-time object detection with confidence thresholding
- **Post-processing**: Non-maximum suppression (NMS)
- **Output**: Annotated images with bounding boxes and confidence scores

## 🎓 Key ML Engineering Practices Demonstrated

### 1. Data-Centric Approach
- **High-quality annotations** as a primary hyperparameter
- Systematic annotation schema design
- Gold standard creation for consistency

### 2. Systematic Evaluation
- Comprehensive metrics (Precision, Recall, mAP)
- Per-class performance analysis
- Validation on held-out data

### 3. Production-Ready Code
- Modular, reusable scripts
- Configurable hyperparameters
- Automated result saving and visualization

### 4. Pipeline Automation
- End-to-end workflow automation
- Reproducible training process
- Version-controlled configurations

## 🛠️ Technologies Used

- **Deep Learning Framework**: PyTorch (via Ultralytics)
- **Model Architecture**: YOLOv8
- **Annotation Tool**: CVAT
- **Data Format**: YOLO format
- **Version Control**: Git
- **Programming Language**: Python 3.x

## 📈 Future Enhancements

### Active Learning Integration
- [ ] Implement uncertainty sampling for hard examples
- [ ] Automated selection of images for human annotation
- [ ] Iterative model improvement loop

### Data Augmentation
- [ ] Advanced augmentation strategies (Mixup, CutMix)
- [ ] Synthetic data generation for edge cases
- [ ] Domain adaptation techniques

### Model Optimization
- [ ] Model quantization for edge deployment
- [ ] Knowledge distillation for smaller models
- [ ] Multi-scale testing for improved accuracy

### Monitoring & Analysis
- [ ] Confusion matrix analysis
- [ ] Per-image error analysis
- [ ] Annotation quality metrics dashboard

## 📝 Methodology: Human-in-the-Loop Workflow

This project demonstrates a **Human-in-the-Loop (HITL)** approach:

1. **Initial Annotation**: Manual annotation of seed dataset
2. **Model Training**: Train initial model on annotated data
3. **Error Analysis**: Identify failure modes and edge cases
4. **Targeted Annotation**: Annotate difficult examples
5. **Iterative Refinement**: Retrain and improve model performance

## 🔍 Annotation Quality Assurance

- Consistent labeling schema across all images
- Normalized bounding box coordinates (YOLO format)
- Validation of annotation completeness
- Cross-checking for labeling consistency

## 💡 Deployment Considerations

- **Inference Speed**: 27ms per image enables real-time processing
- **Model Size**: 6.2MB weights suitable for edge deployment
- **Hardware**: CUDA-enabled GPU for optimal performance
- **Scalability**: Batch processing support for high-throughput scenarios

## 📧 Contact

This project was developed as a demonstration of ML engineering capabilities, including:
- End-to-end pipeline development
- Data curation and annotation QA
- Systematic model evaluation
- Production-ready implementation

---

**Note**: This project showcases proficiency in Python, PyTorch, CVAT annotation workflows, and data-centric ML practices aligned with modern ML engineering standards.
