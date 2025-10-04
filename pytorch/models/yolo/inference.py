import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
try:  # Allow running as module or standalone script
    from .yolov1 import YOLOv1
    from .utils import load_darknet_weights, convert_cell_boxes_to_boxes, non_max_suppression
except ImportError:  # pragma: no cover - fallback when executed as script
    from yolov1 import YOLOv1
    from utils import load_darknet_weights, convert_cell_boxes_to_boxes, non_max_suppression
import yaml

def load_class_names(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image(image_path, input_size=448):
    if isinstance(input_size, (list, tuple)):
        resize_size = tuple(input_size)
    else:
        resize_size = (input_size, input_size)

    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    orig_size = image.size
    image = transform(image).unsqueeze(0)
    return image, orig_size

def draw_boxes(image_path, boxes, class_names, scores, class_preds, conf_threshold=0.5):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # Scale boxes to original image size
    orig_w, orig_h = image.size
    boxes[:, [0, 2]] *= orig_w
    boxes[:, [1, 3]] *= orig_h
    
    # Convert from center format to corner format
    boxes[:, 0] -= boxes[:, 2] / 2  # x1
    boxes[:, 1] -= boxes[:, 3] / 2  # y1
    boxes[:, 2] += boxes[:, 0]      # x2
    boxes[:, 3] += boxes[:, 1]      # y2
    
    for box, score, cls_pred in zip(boxes, scores, class_preds):
        if score > conf_threshold:
            x1, y1, x2, y2 = map(int, box.tolist())
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            
            # Add label
            label = f'{class_names[cls_pred]} {score:.2f}'
            text_w, text_h = draw.textsize(label)
            draw.rectangle([x1, y1, x1 + text_w, y1 + text_h], fill='red')
            draw.text((x1, y1), label, fill='white')
    
    return image

def main():
    # Load config
    config_path = Path(__file__).with_name('config_yolov1.yaml')
    with config_path.open('r') as f:
        config = yaml.safe_load(f)

    model_config = config.get('model', {})
    inference_config = config.get('inference', {})
    data_config = config.get('data', {})

    # Initialize model
    split_size = model_config.get('split_size', 7)
    num_classes = model_config.get('num_classes', 20)
    num_boxes = model_config.get('num_boxes', 2)
    input_size = model_config.get('input_size', [448, 448])

    model = YOLOv1(
        num_classes=num_classes,
        num_boxes=num_boxes,
        num_cells=(split_size, split_size),
    )
    model.eval()

    # Load weights
    weights_path = Path(model_config.get('weights_path', 'weights/yolov1.weights'))
    if not weights_path.is_absolute():
        repo_root = config_path.parents[3] if len(config_path.parents) > 3 else config_path.parent
        resolved = repo_root / weights_path
        if resolved.exists():
            weights_path = resolved
        else:
            weights_path = config_path.parent / weights_path

    load_darknet_weights(model, str(weights_path))

    # Load class names (VOC classes)
    class_names = data_config.get('class_names', [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    # Process image
    image_path = '../../data_input/images/000000000139.jpg'
    image, orig_size = preprocess_image(image_path, input_size=input_size)

    # Get predictions
    with torch.no_grad():
        predictions = model(image)
        
    # Convert predictions to boxes
    boxes, class_preds, scores = convert_cell_boxes_to_boxes(predictions)
    boxes = boxes.squeeze(0)
    class_preds = class_preds.squeeze(0)
    scores = scores.squeeze(0)

    # Apply NMS
    nms_threshold = inference_config.get('nms_threshold', model_config.get('nms_threshold', 0.4))
    nms_indices = non_max_suppression(boxes, scores, iou_threshold=nms_threshold)
    boxes = boxes[nms_indices]
    class_preds = class_preds[nms_indices]
    scores = scores[nms_indices]

    # Draw boxes and save
    conf_threshold = inference_config.get('conf_threshold', model_config.get('confidence_threshold', 0.5))
    annotated_image = draw_boxes(
        image_path,
        boxes,
        class_names,
        scores,
        class_preds,
        conf_threshold=conf_threshold,
    )
    output_path = Path('../../sample_output/yolov1_annotated.jpg')
    output_path.parent.mkdir(exist_ok=True)
    annotated_image.save(output_path)
    print(f'Saved annotated image to {output_path}')

if __name__ == '__main__':
    main()
