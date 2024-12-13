from PIL import Image
import torch
import cv2
import numpy as np
from PIL import ImageDraw, ImageFont
import torchvision.transforms as T

transform = T.Compose([T.ToTensor()])

FONT_PATH = None  # Set this if you have a font file
FONT_SIZE = 20

def predict(model, device, image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    return predictions

def annotate_image(image, predictions, labels, palette):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, size=FONT_SIZE) if FONT_PATH else None

    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        label_idx = label.item()
        if 1 <= label_idx <= len(labels):
            coco_label_name = labels[label_idx - 1]
        else:
            coco_label_name = "unknown"

        box = box.tolist()
        draw.rectangle(box, outline=palette[coco_label_name], width=2)
        draw.text((box[0], box[1]), f"{coco_label_name} ({score:.2f})", fill="white", font=font)
    return image

def process_image(image_path, model, device, labels, palette):
    try:
        image = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, OSError) as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    input_tensor = transform(image).unsqueeze(0)
    predictions = predict(model, device, input_tensor)
    annotated_image = annotate_image(image, predictions, labels, palette)
    return annotated_image

def process_video(video_path, model, device, labels, palette, output_dir, fps=30.0):
    cap = cv2.VideoCapture(video_path)
    output_path = f"{output_dir}/annotated_{os.path.basename(video_path)}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        input_tensor = transform(image).unsqueeze(0)
        predictions = predict(model, device, input_tensor)

        annotated_image = annotate_image(image, predictions, labels, palette)
        annotated_frame = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
        out.write(annotated_frame)

    cap.release()
    out.release()
