# Sphere Detection with Neuronal Networks

## About the Project

This project is part of a Master's program and focuses on detecting spherical objects (balls) using different deep learning approaches.  
The objective is to train, evaluate, and compare various neural network models for sphere detection and benchmark them against a classical computer vision method implemented in a prior Bachelor's thesis.

In the final stage, the developed models will be optimized to run on an NVIDIA Jetson device for real-time inference.

## Project Goals

- **Dataset Preparation:** Validate and complete labeling of an existing dataset.
- **Model Training:** Train a custom YOLO-based model and potentially other architectures.
- **Evaluation:** Measure performance (precision, recall, mAP) and compare results between models.
- **Benchmarking:** Analyze differences between deep learning models and classical vision-based methods.
- **Deployment:** Optimize and deploy the best-performing model on a Jetson device (e.g., Nano, Xavier).

## Technologies Used

- Python
- PyTorch
- YOLO (e.g., YOLOv5, YOLOv8)
- ONNX / TensorRT (for Jetson optimization)
- OpenCV (for classical vision techniques)
- NVIDIA Jetson Platform

## Repository Structure

```plaintext
├── dataset/             # Data and annotations
├── models/              # Trained models and configurations
├── scripts/             # Training, evaluation, and inference scripts
├── jetson_deployment/   # Jetson-specific optimization and deployment
├── results/             # Evaluation results and benchmark comparisons
├── report/              # Documentation and analysis
├── README.md            # Project overview
└── requirements.txt     # Python dependencies
```

## Milestones

- [x] Project setup and GitHub initialization
- [ ] Dataset validation and annotation
- [ ] Model training and tuning
- [ ] Evaluation and comparison
- [ ] Deployment optimization for Jetson
- [ ] Final documentation and report

## Portfolio Value

This project demonstrates expertise in:
- Deep Learning for object detection
- Data preparation and annotation
- Model evaluation and comparison
- Deployment of neural networks on embedded systems (Jetson)

## License

TBD