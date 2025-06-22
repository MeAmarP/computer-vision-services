# **[WIP] PyTorch-TorchVision Computer Vision Service**

A streamlined service to **run and test image classification, object detection, semantic segmentation, and keypoint detection models** available in TorchVision's model zoo. This GPU-enabled service leverages PyTorch and TorchVision to deliver efficient inference for a variety of computer vision tasks on both images and videos.

NOTE: This version currently supports *Object Detection* and *Image Classification* models.
---

## **Features**
- **Pretrained Models**: Supports pretrained models available in [torchvision](https://pytorch.org/vision/stable/models.html#).
- **Multi-Input Support**: Processes both images and videos.
- **Configurable**: Customize input directories, output directories, and detection thresholds via a YAML configuration file.
- **Dockerized**: Easy to deploy with GPU acceleration using NVIDIA Docker.
- **Annotations**: Outputs annotated images/videos with inference output.

## **Getting Started**

### **1. Clone the Repository**
```bash
git clone https://github.com/MeAmarP/computer-vision-services.git
cd computer-vision-services
```

### **2. Prerequisites**

- **NVIDIA GPU with CUDA** (for GPU acceleration)
- **Docker** and **NVIDIA Container Toolkit**
  - Install Docker: [Docker Installation Guide](https://docs.docker.com/get-docker/)
  - Install NVIDIA Container Toolkit: [Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### **Setup**

1. Install [Poetry](https://python-poetry.org/docs/#installation).
2. Install project dependencies:

```bash
poetry install              # base dependencies
poetry install --with dev   # include test dependencies
```

### **Running Tests**

Use the `run_tests.sh` script from the repository root. It wraps `pytest` and writes output to `test-report.md`. The script assumes dependencies were installed via Poetry.


### **3. Directory Structure**
Ensure your project directories are structured like this:
```
computer-vision-services/
|──  pytorch/
|   ├── config.yaml          # Configuration file for input/output directories, model, etc.
|   ├── Dockerfile           # Dockerfile for building the service
|   ├── infer.py             # Image and Video pre & post process for inference
|   ├── main.py              # Main script for inference
|   ├── utils.py             # Utility functions.
|   ├── pyproject.toml       # Poetry configuration
|   ├── data_input/          # Sample input data
|   │   ├── images/          # Place your images here
|   │   └── videos/          # Place your videos here
|   └── infer_output/        # Output directory (auto-created if missing)
```

### **4. Configuration**

Edit the `config.yaml` file to specify the task, input and output directories, model name and detection thresholds:
```yaml
input_dir: "data_input/images"       # Path to input images/videos
output_dir: "infer_output"     # Path to save annotated outputs
task: "object_detection"               # image_classification|semantic_segmentation
model_name: "fasterrcnn_resnet50_fpn"  # Model name
threshold: 0.5                # Confidence threshold for predictions
```


### **5. Build the Docker Image**
Build the Docker image with the following command: Make sure you are inside `pytorch/` directory

```bash
docker build --pull --no-cache -t det-service .
```

### **6. Run the Docker Container**

```bash
docker run --gpus all --shm-size=4g -v $(pwd)/pytorch:/app det-service
```

## **Customization**

### **Switching Models**
To use a different object detection model, update the `model_name` in the `config.yaml` file. 
*Check Supported models [here](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)*

### **Adjusting Detection Thresholds**
Modify the `threshold` in `config.yaml` to change the confidence score cutoff for object detection.


## **Development**

### **Running Locally**
If you want to run the script locally without Docker:
1. Install Python 3.9+.
2. Install [Poetry](https://python-poetry.org/) and project dependencies:
   ```bash
   curl -sSL https://install.python-poetry.org | python3
   poetry install
   ```
3. Run the script using Poetry:
   ```bash
   poetry run python main.py
   ```

---

## **Example Outputs**

### **Input Image**
![Input Example](https://github.com/MeAmarP/computer-vision-services/blob/b01a7e4e5fed2d4fc3021d6369e922144000c8ef/pytorch/data_input/images/1.jpg)

### **Output Image: Object Detection**
![Output Example - Object Detection](https://github.com/MeAmarP/computer-vision-services/blob/c10641371aec2ac353e7276a0369f74a8f728dfc/pytorch/sample_output/fcos_resnet50_fpn_obj_detect_annotated.jpg)

### **Output Image: Semantic Segmentation**
![Output Example - Segmentation](https://github.com/MeAmarP/computer-vision-services/blob/c10641371aec2ac353e7276a0369f74a8f728dfc/pytorch/sample_output/fcn_resnet50_segmentation_annotated.jpg)

### **Output Image: Image Classification**
![Output Example - Classification](https://github.com/MeAmarP/computer-vision-services/blob/c10641371aec2ac353e7276a0369f74a8f728dfc/pytorch/sample_output/mobilenet_v3_large_classification_annotated.jpg)

## Contact
**Author**: [MeAmarP](https://github.com/MeAmarP)
