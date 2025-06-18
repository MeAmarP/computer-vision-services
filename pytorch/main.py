import os
import yaml
import torch
from tqdm import tqdm

from infer import (
    process_image,
    process_video,
    process_image_segmentation,
    process_video_segmentation,
    process_image_classification,
    process_video_classification,
)
from utils import generate_color_palette

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_model_from_config(config: dict, device: torch.device):
    task = config.get("task", "object_detection")
    model_name = config["model_name"]
    model_weights_class_name = config["model_weights_class"]
    model_weights_name = config["model_weights"]
    threshold = config.get("threshold", 0.5)

    if task == "semantic_segmentation":
        import torchvision.models.segmentation as segmentation

        model_fn = getattr(segmentation, model_name)
        weights_class = getattr(segmentation, model_weights_class_name)
        weights = getattr(weights_class, model_weights_name)
        model = model_fn(weights=weights).to(device)
    elif task == "image_classification":
        import torchvision.models as models

        model_fn = getattr(models, model_name)
        weights_class = getattr(models, model_weights_class_name)
        weights = getattr(weights_class, model_weights_name)
        model = model_fn(weights=weights).to(device)
    else:  # object_detection or keypoint_detection
        import torchvision.models.detection as detection

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
    model_name = config["model_name"]
    task = config.get("task", "object_detection")
    infer_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(infer_output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_config(config, device)

    # Load labels from external YAML file based on task
    labels_config_path = config["labels"][task]
    with open(labels_config_path, "r") as f:
        labels_yaml = yaml.safe_load(f)
    labels = labels_yaml[task]

    palette = generate_color_palette(labels) if task != "image_classification" else None
    print("\n")
    print("---"*20)
    print(os.listdir(input_dir))
    print("---"*20)
    print("\n")
    for filename in tqdm(os.listdir(input_dir)):
        input_path = os.path.abspath(os.path.join(input_dir, filename))
        print(f"Processing {input_path}...")
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            if task == "semantic_segmentation":
                annotated_image = process_image_segmentation(
                    input_path, model, device, labels, palette
                )
            elif task == "image_classification":
                annotated_image = process_image_classification(
                    input_path, model, device, labels
                )
            else:
                annotated_image = process_image(
                    input_path, model, device, labels, palette
                )
            if annotated_image:
                annotated_image.save(
                    os.path.join(infer_output_dir, f"annotated_{filename}")
                )
        elif filename.lower().endswith(".mp4"):
            if task == "semantic_segmentation":
                process_video_segmentation(
                    input_path, model, device, labels, palette, infer_output_dir
                )
            elif task == "image_classification":
                process_video_classification(
                    input_path, model, device, labels, infer_output_dir
                )
            else:
                process_video(
                    input_path, model, device, labels, palette, infer_output_dir
                )

if __name__ == "__main__":
    main()
