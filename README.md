# Face Mask Detection System

A real-time face mask detector built with a Convolutional Neural Network (CNN) and OpenCV. The system detects faces in a live webcam feed using a Haar cascade classifier, then classifies each detected face as **MASK** or **NO MASK** using a trained Keras model.

## How It Works

1. **Training (`TrainModelInit.py`)** — A CNN is trained from scratch on a labeled dataset of masked and unmasked face images (`Dataset/train` and `Dataset/test`), using data augmentation (shear, zoom, horizontal flip) via `ImageDataGenerator`. The trained model is saved as `maskmodel.h5`.
2. **Inference (`FaceMaskInit.py`)** — Opens the webcam feed, detects faces frame-by-frame using `haarcascade_frontalface_default.xml`, crops each face, runs it through the trained model, and overlays a bounding box + label (green "MASK" / red "NO MASK") along with a timestamp.

## Model Architecture

A simple 3-block CNN:
- 3x `Conv2D` + `MaxPooling2D` layers (32 filters each, ReLU activation)
- `Flatten` → `Dense(100, relu)` → `Dense(1, sigmoid)`
- Optimizer: Adam, Loss: Binary Crossentropy

## Results

Training was run for 20–25 epochs, reaching ~97–99% training accuracy with test accuracy tracking closely (see `accuracy.png`, `accuracy1.png`, `loss.png`, `loss1.png` for training curves from different runs).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train the model:**
```bash
python TrainModelInit.py
```

**Run real-time detection (requires a webcam and a trained `maskmodel.h5`):**
```bash
python FaceMaskInit.py
```
Press `q` to quit the live detection window.

## Project Structure

```
Face Mask Detection System/
├── Dataset/
│   ├── train/
│   │   ├── with_mask/
│   │   └── without_mask/
│   └── test/
│       ├── with_mask/
│       └── without_mask/
├── TrainModelInit.py          # Trains the CNN and saves maskmodel.h5
├── FaceMaskInit.py            # Real-time webcam mask detection
├── haarcascade_frontalface_default.xml  # OpenCV pretrained face detector
├── maskmodel.h5                # Trained model weights
├── requirements.txt
└── README.md
```

## Tech Stack

- Python
- TensorFlow / Keras — CNN training and inference
- OpenCV — face detection (Haar cascade) and video I/O
- NumPy, Matplotlib — data handling and result visualization

## Notes

- The dataset is not included in this repository (see `.gitignore`) due to size. Structure your own dataset under `Dataset/train/{with_mask,without_mask}` and `Dataset/test/{with_mask,without_mask}` to retrain.
- `temp.jpg` is a scratch file written on every detection frame and is git-ignored.
