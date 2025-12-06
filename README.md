# Bitirme Project

This project implements a sensor fusion model for 3D object detection, designed to work with the KITTI dataset. It combines image data and point cloud data using a deep learning approach inspired by PointFusion.

## Features

- **Sensor Fusion**: Combines Camera (ResNet/YOLOv5) and LiDAR (PointNet) features.
- **Configurable Backbone**: Choose between ResNet50 and YOLOv5 for image feature extraction.
- **Unsupervised Learning**: Uses an unsupervised loss function for training.
- **Validation & Logging**: Integrated validation loop and WandB logging.
- **Inference**: Script to run inference on new samples.

## Project Structure

- **Backbone/**: Neural network models (`pointfusion.py`, `Pointnet/`, `Yolov5/`).
- **Config/**: Configuration files (`train_test.yaml`, `kitti_config.py`).
- **DataProcess/**: Data loading and preprocessing (`kitti_dataset.py`, `kitti_utils.py`).
- **Utils/**: Visualization and utility functions.
- **train.py**: Main training script.
- **inference.py**: Inference script.

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository.
2. Install dependencies.
3. Prepare the KITTI dataset.

## Configuration

Configure the project in `Config/train_test.yaml`:

```yaml
dataset:
  root_dir: /path/to/kitti/dataset
  backbone: resnet # or yolov5

train:
  wandb:
    use: true
    project: bitirme_project
```

## Usage

### Training

```bash
python train.py
```

### Inference

```bash
python inference.py --checkpoint checkpoints/best_model.pth --visualize
```
