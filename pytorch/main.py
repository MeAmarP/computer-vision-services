import os
import yaml
import torch
from tqdm import tqdm

from infer import process_image, process_video
from utils import generate_color_palette

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_model_from_config(config: dict, device: torch.device):
    import torchvision.models.detection as detection

    model_name = config["model_name"]  # e.g. "fasterrcnn_resnet50_fpn_v2"
    model_weights_class_name = config["model_weights_class"]  # e.g. "FasterRCNN_ResNet50_FPN_V2_Weights"
    model_weights_name = config["model_weights"]  # e.g. "COCO_V1"
    threshold = config["threshold"]

    # Dynamically get the model function, weights class, and specified weights
    model_fn = getattr(detection, model_name)
    weights_class = getattr(detection, model_weights_class_name)
    weights = getattr(weights_class, model_weights_name)

    model = model_fn(weights=weights, score_thresh=threshold).to(device)
    model.eval()
    return model

def main():
    config = load_config("config.yaml")
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    labels = config["coco_labels"]
    model_name = config["model_name"]
    infer_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(infer_output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_config(config, device)

    palette = generate_color_palette(labels)

    for filename in tqdm(os.listdir(input_dir)):
        input_path = os.path.join(input_dir, filename)
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            annotated_image = process_image(input_path, model, device, labels, palette)
            if annotated_image:

                annotated_image.save(os.path.join(infer_output_dir, f"annotated_{filename}"))
        elif filename.lower().endswith(".mp4"):
            process_video(input_path, model, device, labels, palette, infer_output_dir)

if __name__ == "__main__":
    main()
