import os
from PIL import Image
import cv2
import torch
import numpy as np
from PIL import ImageDraw, ImageFont
from tqdm import tqdm
import logging

import torchvision.transforms as T
from torchvision.io import read_video, write_video

logger = logging.getLogger(__name__)

transform = T.Compose([T.ToTensor()])

# Transform pipeline used for image classification models
classification_transform = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

FONT_PATH = None  # Set this if you have a font file
FONT_SIZE = 20

def predict(model, device, image_tensor):
    """Run model prediction on a tensor."""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    logger.debug("Predictions: %s", predictions)
    return predictions

def annotate_image(image, predictions, labels, palette):
    """Annotate detection results on an image."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, size=FONT_SIZE) if FONT_PATH else None

    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        label_idx = label.item()
        if 0 <= label_idx < len(labels):
            coco_label_name = labels[label_idx]
        else:
            coco_label_name = "unknown"

        logger.debug("Box: %s, Label: %s, Score: %.2f", box.tolist(), coco_label_name, score)

        box = box.tolist()
        draw.rectangle(box, outline=palette[coco_label_name], width=2)
        draw.text((box[0], box[1]), f"{coco_label_name} ({score:.2f})", fill="white", font=font)
    return image

def annotate_instance_segmentation(image, predictions, labels, palette, alpha=0.4):
    """Annotate instance segmentation results on an image."""
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(FONT_PATH, size=FONT_SIZE) if FONT_PATH else None

    for box, mask, label, score in zip(
        predictions["boxes"],
        predictions.get("masks", torch.zeros(0)),
        predictions["labels"],
        predictions["scores"],
    ):
        label_idx = label.item()
        label_name = labels[label_idx] if 0 <= label_idx < len(labels) else "unknown"
        color = palette.get(label_name, (255, 255, 255))

        mask_np = (mask[0] > 0.5).byte().cpu().numpy() * 255
        mask_img = Image.fromarray(mask_np, mode="L")
        colored = Image.new("RGBA", image.size, color + (int(255 * alpha),))
        overlay.paste(colored, (0, 0), mask_img)

        box = box.tolist()
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0], box[1]), f"{label_name} ({score:.2f})", fill="white", font=font)

    blended = Image.alpha_composite(image, overlay)
    return blended.convert("RGB")

def annotate_keypoints(image, predictions, labels, palette):
    """Annotate keypoint detection results on an image."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, size=FONT_SIZE) if FONT_PATH else None

    for box, kpts, label, score in zip(
        predictions["boxes"],
        predictions.get("keypoints", torch.zeros(0)),
        predictions["labels"],
        predictions["scores"],
    ):
        label_idx = label.item()
        label_name = labels[label_idx] if 0 <= label_idx < len(labels) else "unknown"
        color = palette.get(label_name, (255, 255, 255))

        box = box.tolist()
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0], box[1]), f"{label_name} ({score:.2f})", fill="white", font=font)

        for kp in kpts:
            x, y, v = kp.tolist()
            if v > 0:
                r = 3
                draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    return image

def process_image(image_path, model, device, labels, palette):
    """Run detection on a single image file."""
    try:
        image = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, OSError) as e:
        logger.error("Error opening image %s: %s", image_path, e)
        return None

    input_tensor = transform(image).unsqueeze(0)
    predictions = predict(model, device, input_tensor)
    annotated_image = annotate_image(image, predictions, labels, palette)
    return annotated_image


def process_image_instance_segmentation(image_path, model, device, labels, palette):
    """Run instance segmentation on a single image."""
    try:
        image = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, OSError) as e:
        logger.error("Error opening image %s: %s", image_path, e)
        return None

    input_tensor = transform(image).unsqueeze(0)
    predictions = predict(model, device, input_tensor)
    annotated_image = annotate_instance_segmentation(image, predictions, labels, palette)
    return annotated_image


def process_image_keypoint(image_path, model, device, labels, palette):
    """Run keypoint detection on a single image."""
    try:
        image = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, OSError) as e:
        logger.error("Error opening image %s: %s", image_path, e)
        return None

    input_tensor = transform(image).unsqueeze(0)
    predictions = predict(model, device, input_tensor)
    annotated_image = annotate_keypoints(image, predictions, labels, palette)
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
    logger.info("Processing video %s", video_path)
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Error opening video file %s", video_path)
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    logger.debug("Frame width: %s height: %s input_fps: %s", frame_width, frame_height, input_fps)
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
    logger.info("Annotated video saved to %s", output_path)


def process_video_instance_segmentation(video_path, model, device, labels, palette, infer_output_dir, fps=30):
    """Process a video for instance segmentation."""
    output_path = os.path.join(infer_output_dir, f"annotated_{os.path.basename(video_path)}")
    os.makedirs(infer_output_dir, exist_ok=True)
    logger.info("Processing instance segmentation video %s", video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Error opening video file %s", video_path)
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
        input_tensor = transform(frame_image).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = model(input_tensor)[0]

        annotated_image = annotate_instance_segmentation(frame_image, predictions, labels, palette)
        annotated_frame = np.array(annotated_image)
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        out.write(annotated_frame_bgr)

    cap.release()
    out.release()
    logger.info("Annotated video saved to %s", output_path)


def process_video_keypoint(video_path, model, device, labels, palette, infer_output_dir, fps=30):
    """Process a video for keypoint detection."""
    output_path = os.path.join(infer_output_dir, f"annotated_{os.path.basename(video_path)}")
    os.makedirs(infer_output_dir, exist_ok=True)
    logger.info("Processing keypoint detection video %s", video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Error opening video file %s", video_path)
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
        input_tensor = transform(frame_image).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = model(input_tensor)[0]

        annotated_image = annotate_keypoints(frame_image, predictions, labels, palette)
        annotated_frame = np.array(annotated_image)
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        out.write(annotated_frame_bgr)

    cap.release()
    out.release()
    logger.info("Annotated video saved to %s", output_path)


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
    blended = Image.blend(image, mask_image, alpha)
    blended = blended.convert("RGB")  # Convert back to RGB for consistency
    logger.debug("Annotated segmentation with alpha %s", alpha)
    return blended


def process_image_segmentation(image_path, model, device, labels, palette):
    try:
        image = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, OSError) as e:
        logger.error("Error opening image %s: %s", image_path, e)
        return None

    input_tensor = transform(image).unsqueeze(0)
    output = predict_segmentation(model, device, input_tensor)
    return annotate_segmentation(image, output, labels, palette)


def process_video_segmentation(video_path, model, device, labels, palette, infer_output_dir, fps=30):
    output_path = os.path.join(infer_output_dir, f"annotated_{os.path.basename(video_path)}")
    os.makedirs(infer_output_dir, exist_ok=True)
    logger.info("Processing segmentation video %s", video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Error opening video file %s", video_path)
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
    logger.info("Annotated video saved to %s", output_path)


def annotate_classification(image, probs, labels, topk=5):
    """Annotate an image with classification results."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, size=FONT_SIZE) if FONT_PATH else None
    top_probs, top_ids = torch.topk(probs, k=min(topk, probs.shape[0]))
    for i, (p, idx) in enumerate(zip(top_probs, top_ids)):
        label_idx = idx.item()
        label_name = labels[label_idx] if labels and label_idx < len(labels) else str(label_idx)
        draw.text((10, 10 + i * FONT_SIZE), f"{label_name}: {p:.2f}", fill="white", font=font)
    return image


def process_image_classification(image_path, model, device, labels):
    """Run image classification on a single image."""
    try:
        image = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, OSError) as e:
        logger.error("Error opening image %s: %s", image_path, e)
        return None

    input_tensor = classification_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    annotated_image = annotate_classification(image, probs, labels, topk=1)
    return annotated_image


def process_video_classification(video_path, model, device, labels, infer_output_dir, fps=30):
    """Run classification on each frame of a video."""
    output_path = os.path.join(infer_output_dir, f"annotated_{os.path.basename(video_path)}")
    os.makedirs(infer_output_dir, exist_ok=True)
    logger.info("Processing classification video %s", video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Error opening video file %s", video_path)
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
        input_tensor = classification_transform(frame_image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

        annotated_image = annotate_classification(frame_image, probs, labels, topk=1)
        annotated_frame = np.array(annotated_image.convert("RGB"))
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        out.write(annotated_frame_bgr)

    cap.release()
    out.release()
    logger.info("Annotated video saved to %s", output_path)


