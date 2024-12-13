import random
import os
import yaml
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image, ImageDraw
import cv2

from tqdm import tqdm


def load_config(config_path: str) -> dict:
    """
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def main():
    config = load_config("config.yaml")
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    coco_labels = config["coco_labels"]
    box_thresh = config["threshold"]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # * Use GPU for Inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load pre-trained object detection model
weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=box_thresh)
model.to(device)
model.eval()

# Define image transformation
transform = T.Compose([
    T.ToTensor()
])

# Generate unique colors for each class label
def generate_color_palette(labels):
    random.seed(42)
    palette = {label: tuple(random.randint(0, 255) for _ in range(3)) for label in labels}
    palette["unknown"] = (255,255,255)
    return palette

color_palette = generate_color_palette(coco_labels)

# ! TODO --> Implement Logging
# ! TODO --> Define a predict function that can be used for both images and videos
# ! TODO --> Implement batch processing for images and videos
def predict(img: Image.Image):
    pass

def process_image(image_path):
    # TODO --> DO exception handling here with IO operation
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        predictions = model(input_tensor)[0]

    draw = ImageDraw.Draw(image)
    for box, label, score in zip(
        predictions['boxes'],
        predictions['labels'],
        predictions['scores']
    ):
        coco_label_name = "unknown"  if label > len(coco_labels) else coco_labels[label-1]
        draw.rectangle(box.tolist(), outline=color_palette[coco_label_name], width=2)
        draw.text((box[0], box[1]), f"{coco_label_name} ({score:.2f})", fill="white", font_size=15)

    return image

def process_video(video_path):
    pass
    # cap = cv2.VideoCapture(video_path)
    # output_path = os.path.join(output_dir, f"annotated_{os.path.basename(video_path)}")
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     image = Image.fromarray(frame_rgb)
    #     annotated_image = predict(image)_
    # !    annotated_frame = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
    #     out.write(annotated_frame)

    # cap.release()
    # out.release()

def main():
    for filename in tqdm(os.listdir(input_dir)):
        input_path = os.path.join(input_dir, filename)
        if filename.endswith((".jpg", ".png")):
            annotated_image = process_image(input_path)
            annotated_image.save(os.path.join(output_dir, f"annotated_{filename}"))
        elif filename.endswith(".mp4"):
            process_video(input_path)

if __name__ == "__main__":
    main()
