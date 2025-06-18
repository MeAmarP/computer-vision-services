import os
from PIL import Image
import cv2
import torch
import numpy as np
from PIL import ImageDraw, ImageFont
from tqdm import tqdm

import torchvision.transforms as T
from torchvision.io import read_video, write_video

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



def process_video(video_path, model, device, labels, palette, infer_output_dir, fps=30):
    """
    Processes a video for object detection and creates an annotated video using OpenCV.

    Args:
        video_path (str): Path to the input video.
        model (torch.nn.Module): The object detection model.
        device (torch.device): Device for computation (CPU or GPU).
        labels (dict): Class labels mapping.
        palette (dict): Color palette for annotations.
        infer_output_dir (str): Directory to save annotated video.
        fps (float): Frames per second for the output video.
    """
    # Define output path
    output_path = os.path.join(infer_output_dir, f"annotated_{os.path.basename(video_path)}")
    os.makedirs(infer_output_dir, exist_ok=True)
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f" Frame width: {frame_width}\n Frame height: {frame_height}\n Input FPS: {input_fps}")
    fps = input_fps if fps is None else fps

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Process each frame
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(frame_count), desc="Processing frames", colour="green"):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB (OpenCV loads in BGR format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frame to PIL image
        frame_image = Image.fromarray(frame_rgb)

        # Convert the image to a tensor and perform inference
        input_tensor = transform(frame_image).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = model(input_tensor)[0]

        # Annotate the frame
        annotated_image = annotate_image(frame_image, predictions, labels, palette)

        # Convert annotated image back to NumPy array
        annotated_frame = np.array(annotated_image)

        # Convert back to BGR for OpenCV and write to video
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        out.write(annotated_frame_bgr)

    # Release video objects
    cap.release()
    out.release()
    print(f"Annotated video saved to {output_path}")


def predict_segmentation(model, device, image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        if isinstance(output, dict) and "out" in output:
            output = output["out"]
    return output[0]


def annotate_segmentation(image, output, labels, palette, alpha=0.6):
    class_map = output.argmax(0).byte().cpu().numpy()
    color_mask = np.zeros((class_map.shape[0], class_map.shape[1], 3), dtype=np.uint8)
    for idx, label in enumerate(labels):
        mask = class_map == idx
        color_mask[mask] = palette.get(label, (255, 255, 255))
    mask_image = Image.fromarray(color_mask)
    image = image.convert("RGBA")
    mask_image = mask_image.convert("RGBA")
    return Image.blend(image, mask_image, alpha)


def process_image_segmentation(image_path, model, device, labels, palette):
    try:
        image = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, OSError) as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    input_tensor = transform(image).unsqueeze(0)
    output = predict_segmentation(model, device, input_tensor)
    return annotate_segmentation(image, output, labels, palette)


def process_video_segmentation(video_path, model, device, labels, palette, infer_output_dir, fps=30):
    output_path = os.path.join(infer_output_dir, f"annotated_{os.path.basename(video_path)}")
    os.makedirs(infer_output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = input_fps if fps is None else fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(frame_count), desc="Processing frames", colour="green"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)
        input_tensor = transform(frame_image).unsqueeze(0)
        output = predict_segmentation(model, device, input_tensor)
        annotated_image = annotate_segmentation(frame_image, output, labels, palette)
        annotated_frame = np.array(annotated_image.convert("RGB"))
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        out.write(annotated_frame_bgr)

    cap.release()
    out.release()
    print(f"Annotated video saved to {output_path}")


