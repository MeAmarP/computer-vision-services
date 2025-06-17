# **[WIP] PyTorch-TorchVision Computer Vision Service**

A streamlined service to **run and test image classification, object detection, semantic segmentation, and keypoint detection models** available in TorchVision's model zoo. This GPU-enabled service leverages PyTorch and TorchVision to deliver efficient inference for a variety of computer vision tasks on both images and videos.

NOTE: This version currently supports *Object Detection* models.
---

## **Features**
- **Pretrained Models**: Supports pretrained models available in [torchvision](https://pytorch.org/vision/stable/models.html#).
- **Multi-Input Support**: Processes both images and videos.
- **Configurable**: Customize input directories, output directories, detection thresholds, and supported labels via a YAML configuration file.
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
|   ├── myenv.yml            # Python Conda env file
|   ├── data_input/          # Sample input data
|   │   ├── images/          # Place your images here
|   │   └── videos/          # Place your videos here
|   └── infer_output/        # Output directory (auto-created if missing)
```

### **4. Configuration**

Edit the `config.yaml` file to specify input and output directories, model name, detection thresholds, and supported class labels:
```yaml
input_dir: "data_input/images"       # Path to input images/videos
output_dir: "infer_output"     # Path to save annotated outputs
model_name: "fasterrcnn_resnet50_fpn"  # Object detection model
threshold: 0.5                # Confidence threshold for predictions
coco_labels:             # Labels to include in the output
  - person
  - car
  - bicycle
  - dog
  - cat
  - truck
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
2. Install dependencies:
   ```bash
   conda env create -f myenv.yml
   ```
3. Activate conda env
   ```bash
   conda activate myenv
   ```
4. Run the script:
   ```bash
   python main.py
   ```

---

## **Example Outputs**

### **Input Image**
![Input Example](https://github.com/MeAmarP/computer-vision-services/blob/b01a7e4e5fed2d4fc3021d6369e922144000c8ef/pytorch/data_input/images/1.jpg)

### **Output Image**
![Output Example](https://github.com/MeAmarP/computer-vision-services/blob/8ef1a564b7e5d1bf1b689bccb2784eba294719eb/pytorch/infer_output/annotated_1.jpg)

### **Output Video**
![Output Example](https://github.com/MeAmarP/computer-vision-services/blob/036baf8aa925a927e9abd3635dd06626b2a99657/pytorch/infer_output/fcos_resnet50_fpn/annotated_palace.mp4)


## Contact
**Author**: [MeAmarP](https://github.com/MeAmarP)
