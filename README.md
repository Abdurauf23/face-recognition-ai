# Emotion Recognition AI

A deep learning-based facial emotion recognition system built with PyTorch and FastAPI. Upload an image and the model will detect and classify the emotion displayed on the face.

## Overview

This project uses a fine-tuned ResNet-152 model to recognize 7 different emotions from facial images. The application follows a medallion architecture (Bronze/Silver/Gold layers) for data processing and provides a web interface for easy interaction.

## Features

- Real-time emotion recognition from uploaded images
- 7 emotion classes: angry, disgust, fear, happy, neutral, sad, surprise
- Web-based interface for image upload
- Dockerized deployment
- Prediction logging to CSV
- GPU acceleration support (CUDA)

## Architecture

The project follows a medallion data architecture:

- **Bronze Layer**: Raw uploaded images
- **Silver Layer**: Preprocessed and transformed images
- **Gold Layer**: Final predictions with annotated images and CSV logs

## Tech Stack

- **Deep Learning**: PyTorch, TorchVision
- **Model**: ResNet-152 (fine-tuned)
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Jinja2 Templates
- **Containerization**: Docker
- **Image Processing**: Pillow, Matplotlib

## Installation

### Using Docker (Recommended)

**Windows:**
```bash
run_docker.bat
```

**Linux/Mac:**
```bash
chmod +x run_docker.sh
./run_docker.sh
```

### Manual Setup

1. Install Python 3.12.7
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## Usage

1. Navigate to `http://localhost:8000/emotion-recognition/`
2. Upload an image containing a face
3. View the prediction result with confidence percentage

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/emotion-recognition/` | Web interface for image upload |
| POST | `/emotion-recognition/` | Upload image and get prediction |
| GET | `/resources/golden/{filename}` | Retrieve processed images |

## Model Details

- **Base Model**: ResNet-152 (pretrained on ImageNet)
- **Input**: Grayscale images resized to 256x256
- **Output**: 7 emotion classes with confidence scores
- **Model File**: `model_emotion_v3.pth`

## Project Structure

```
face-recognition-ai/
├── main.py              # FastAPI application
├── gold_layer.py        # Emotion recognition model
├── transform_photo.py   # Image preprocessing
├── model_emotion_v3.pth # Trained model weights
├── templates/           # HTML templates
├── Dockerfile           # Docker configuration
├── requirements.txt     # Python dependencies
└── README.md
```

## Requirements

- Python 3.12.7
- CUDA-compatible GPU (optional, for faster inference)
- ~500MB disk space for model weights
