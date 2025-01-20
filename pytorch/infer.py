import os
from PIL import Image
import cv2
import torch
import numpy as np
from PIL import ImageDraw, ImageFont
from tqdm import tqdm

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

# def process_video(video_path, model, device, labels, palette, infer_output_dir, fps=30):
#     """
#     Processes a video for object detection and creates an annotated video using TorchVision utilities.

#     Args:
#         video_path (str): Path to the input video.
#         model (torch.nn.Module): The object detection model.
#         device (torch.device): Device for computation (CPU or GPU).
#         labels (dict): Class labels mapping.
#         palette (dict): Color palette for annotations.
#         infer_output_dir (str): Directory to save annotated video.
#         fps (float): Frames per second for the output video.
#     """
# from torchvision.io import read_video, write_video
#     # Define output path
#     output_path = os.path.join(infer_output_dir, f"annotated_{os.path.basename(video_path)}")

#     # Read video frames and metadata
#     video_frames, _, info = read_video(video_path, pts_unit="sec")
#     # width, height = video_frames.shape[2], video_frames.shape[1]
#     annotated_frames = []

#     for i, frame in enumerate(tqdm(video_frames, colour='green')):
#         # Convert frame (Torch tensor) to a PIL image
#         frame_image = Image.fromarray(frame.numpy())

#         # Convert the image to a tensor and perform inference
#         input_tensor = transform(frame_image).unsqueeze(0).to(device)
#         with torch.no_grad():
#             predictions = model(input_tensor)[0]

#         # Annotate the frame
#         annotated_image = annotate_image(frame_image, predictions, labels, palette)

#         # Convert the annotated frame back to a Torch tensor
#         annotated_frame = torch.from_numpy(np.array(annotated_image, dtype=np.uint8)).to(dtype=torch.uint8)
#         annotated_frames.append(annotated_frame)

#     # Stack frames and write the output video
#     annotated_video = torch.stack(annotated_frames)
#     write_video(output_path, annotated_video, fps=fps, video_codec="h264", options={"crf": "20"})
