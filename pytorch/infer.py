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
LOADED_FONT = None # New global for cached font

def get_font():
    global LOADED_FONT
    if FONT_PATH and LOADED_FONT is None:
        try:
            LOADED_FONT = ImageFont.truetype(FONT_PATH, size=FONT_SIZE)
            logger.debug(f"Successfully loaded font: {FONT_PATH}")
        except IOError: # Catching IOError as per PIL documentation for truetype
            logger.error(f"Could not load font from {FONT_PATH}. PIL will use a default font.")
            # LOADED_FONT remains None, ImageDraw.text will use PIL's default if font is None
    return LOADED_FONT

def predict(model, device, image_tensor):
    """Run model prediction on a tensor."""
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    logger.debug("Predictions: %s", predictions)
    return predictions

def annotate_image(image, predictions, labels, palette):
    """Annotate detection results on an image."""
    draw = ImageDraw.Draw(image)
    font = get_font()

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


# Helper Frame Processing Functions
def _process_frame_detection(frame_rgb_numpy, model, device, labels, palette, **kwargs):
    """Processes a single frame for object detection."""
    # Convert NumPy HWC [0,255] to Tensor CHW [0,1.0]
    tensor_chw = torch.from_numpy(frame_rgb_numpy.transpose((2, 0, 1))).float().div(255)
    input_tensor = tensor_chw.unsqueeze(0).to(device)
    # Perform inference
    with torch.no_grad():
        predictions = model(input_tensor)[0]
    # Convert NumPy array to PIL Image for annotation
    pil_image = Image.fromarray(frame_rgb_numpy)
    # Annotate
    annotated_pil_image = annotate_image(pil_image, predictions, labels, palette)
    return annotated_pil_image

def _process_frame_segmentation(frame_rgb_numpy, model, device, labels, palette, **kwargs):
    """Processes a single frame for semantic segmentation."""
    # Convert NumPy HWC [0,255] to Tensor CHW [0,1.0]
    tensor_chw = torch.from_numpy(frame_rgb_numpy.transpose((2, 0, 1))).float().div(255)
    input_tensor = tensor_chw.unsqueeze(0).to(device)
    # Perform inference
    seg_output = predict_segmentation(model, device, input_tensor) # predict_segmentation handles no_grad
    # Convert NumPy array to PIL Image for annotation
    pil_image = Image.fromarray(frame_rgb_numpy)
    # Annotate
    annotated_pil_image = annotate_segmentation(pil_image, seg_output, labels, palette)
    return annotated_pil_image

def _process_frame_classification(frame_rgb_numpy, model, device, labels, custom_transform_fn, **kwargs):
    """Processes a single frame for image classification."""
    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(frame_rgb_numpy)
    # Apply transforms
    input_tensor = custom_transform_fn(pil_image).unsqueeze(0).to(device)
    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
    # Annotate
    annotated_pil_image = annotate_classification(pil_image, probs, labels, topk=1)
    return annotated_pil_image

# Generic Video Processing Backbone
def _process_video_generic(video_path, model, device, labels, infer_output_dir,
                           frame_processor_callable, process_specific_args, desired_fps=None):
    """
    Generic backbone for processing videos.
    """
    output_filename = f"annotated_{os.path.basename(video_path)}"
    output_path = os.path.join(infer_output_dir, output_filename)
    # os.makedirs(infer_output_dir, exist_ok=True) # Ensure this is called by public functions

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Error opening video file %s", video_path)
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.debug("Video properties: Width=%d, Height=%d, FPS=%.2f", frame_width, frame_height, input_fps)

    actual_fps = input_fps if desired_fps is None else desired_fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, actual_fps, (frame_width, frame_height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0: # Handle cases where frame count is not available or video is empty
        logger.warning("Cannot determine frame count or video is empty: %s", video_path)
        # Try to process at least one frame if possible, or simply return
        # For now, let's assume if frame_count is 0, we can't process.
        cap.release()
        out.release()
        if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
             os.remove(output_path) # Clean up empty file
        return

    for i in tqdm(range(frame_count), desc=f"Processing {os.path.basename(video_path)}", colour="cyan"):
        ret, frame_bgr = cap.read()
        if not ret:
            logger.warning("Failed to read frame %d from %s", i, video_path)
            break

        frame_rgb_numpy = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        try:
            annotated_pil = frame_processor_callable(
                frame_rgb_numpy, model, device, labels, **process_specific_args
            )
            annotated_bgr_numpy = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)
            out.write(annotated_bgr_numpy)
        except Exception as e:
            logger.error("Error processing frame %d in %s: %s", i, video_path, e, exc_info=True)
            # Optionally, write the original frame to keep video length consistent or skip
            # For now, we skip writing this frame.
            pass


    cap.release()
    out.release()
    logger.info("Annotated video saved to %s", output_path)


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
    logger.info("Processing video %s for object detection", os.path.basename(video_path))
    os.makedirs(infer_output_dir, exist_ok=True)
    specific_args = {'palette': palette}
    _process_video_generic(video_path, model, device, labels, infer_output_dir,
                           _process_frame_detection, specific_args, desired_fps=fps)


def predict_segmentation(model, device, image_tensor):
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
    """Processes a video for semantic segmentation using the generic backbone."""
    logger.info("Processing video %s for semantic segmentation", os.path.basename(video_path))
    os.makedirs(infer_output_dir, exist_ok=True)
    specific_args = {'palette': palette}
    _process_video_generic(video_path, model, device, labels, infer_output_dir,
                           _process_frame_segmentation, specific_args, desired_fps=fps)


def annotate_classification(image, probs, labels, topk=5):
    """Annotate an image with classification results."""
    draw = ImageDraw.Draw(image)
    font = get_font()
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
    """Runs classification on each frame of a video using the generic backbone."""
    logger.info("Processing video %s for image classification", os.path.basename(video_path))
    os.makedirs(infer_output_dir, exist_ok=True)
    # classification_transform is a global in infer.py
    specific_args = {'custom_transform_fn': classification_transform}
    _process_video_generic(video_path, model, device, labels, infer_output_dir,
                           _process_frame_classification, specific_args, desired_fps=fps)


