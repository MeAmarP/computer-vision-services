import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

def load_darknet_weights(model, weights_path: str) -> None:
    """
    Load original darknet weights file
    
    Args:
        model: YOLOv1 PyTorch model
        weights_path (str): Path to darknet .weights file
    """
    # Read binary file
    with open(weights_path, 'rb') as f:
        header = np.fromfile(f, dtype=np.int32, count=5)
        if header.size < 5:
            raise ValueError("Invalid Darknet weights file header")
        weights = np.fromfile(f, dtype=np.float32)

    ptr = 0
    for m in model.features.modules():
        if isinstance(m, torch.nn.Conv2d):
            conv = m
            num_w = conv.weight.numel()
            
            if conv.bias is not None:
                num_b = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv.bias)
                ptr += num_b
                conv.bias.data.copy_(conv_b)

            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv.weight)
            ptr += num_w
            conv.weight.data.copy_(conv_w)
    
    # Load detection head weights
    for m in model.head.modules():
        if isinstance(m, torch.nn.Conv2d):
            conv = m
            num_w = conv.weight.numel()
            
            if conv.bias is not None:
                num_b = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv.bias)
                ptr += num_b
                conv.bias.data.copy_(conv_b)

            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv.weight)
            ptr += num_w
            conv.weight.data.copy_(conv_w)

        elif isinstance(m, torch.nn.Linear):
            num_w = m.weight.numel()
            num_b = m.bias.numel()
            
            lin_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(m.bias)
            ptr += num_b
            lin_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(m.weight)
            ptr += num_w

            m.weight.data.copy_(lin_w)
            m.bias.data.copy_(lin_b)
    
    if ptr != len(weights):
        logger.warning(
            "Not all Darknet weights were consumed (used %s of %s)", ptr, len(weights)
        )

def convert_cell_boxes_to_boxes(predictions: torch.Tensor, S: int = 7) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert YOLO grid cell predictions to bounding boxes
    
    Args:
        predictions (torch.Tensor): Raw predictions from YOLOv1 model
        S (int): Grid size (S x S)
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: boxes, class_pred, scores
    """
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, -1)
    
    # Get box coordinates
    box1_xy = torch.sigmoid(predictions[..., :2])
    box1_wh = predictions[..., 2:4]
    box1_conf = torch.sigmoid(predictions[..., 4])
    
    box2_xy = torch.sigmoid(predictions[..., 5:7])
    box2_wh = predictions[..., 7:9]
    box2_conf = torch.sigmoid(predictions[..., 9])
    
    # Get class probabilities
    class_probs = predictions[..., 10:]
    class_probs = torch.softmax(class_probs, dim=-1)
    
    # Convert cell relative coordinates to image coordinates
    cell_indices = torch.arange(S).repeat(batch_size, S, 1).to(predictions.device)
    x = (box1_xy[..., 0].unsqueeze(-1) + cell_indices) / S
    y = (box1_xy[..., 1].unsqueeze(-1) + cell_indices.transpose(1, 2)) / S
    w = box1_wh[..., 0].unsqueeze(-1)
    h = box1_wh[..., 1].unsqueeze(-1)
    
    boxes1 = torch.stack([x, y, w, h], dim=-1)
    
    x = (box2_xy[..., 0].unsqueeze(-1) + cell_indices) / S
    y = (box2_xy[..., 1].unsqueeze(-1) + cell_indices.transpose(1, 2)) / S
    w = box2_wh[..., 0].unsqueeze(-1)
    h = box2_wh[..., 1].unsqueeze(-1)
    
    boxes2 = torch.stack([x, y, w, h], dim=-1)
    
    # Choose box with higher confidence
    boxes = torch.where(box1_conf > box2_conf, boxes1, boxes2)
    scores = torch.max(box1_conf, box2_conf)
    class_pred = torch.argmax(class_probs, dim=-1)
    
    return boxes, class_pred, scores

def non_max_suppression(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5
) -> List[int]:
    """
    Perform Non-Maximum Suppression on boxes
    
    Args:
        boxes (torch.Tensor): Bounding boxes (x, y, w, h)
        scores (torch.Tensor): Confidence scores for each box
        iou_threshold (float): IoU threshold for NMS
        
    Returns:
        List[int]: Indices of boxes to keep
    """
    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
            
        i = order[0]
        keep.append(i)
        
        # Compute IoU with remaining boxes
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= iou_threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
        
    return keep

def intersection_over_union(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two sets of boxes
    
    Args:
        boxes1 (torch.Tensor): First set of boxes (x, y, w, h)
        boxes2 (torch.Tensor): Second set of boxes (x, y, w, h)
        
    Returns:
        torch.Tensor: IoU values
    """
    # Convert to x1, y1, x2, y2
    b1_x1 = boxes1[..., 0] - boxes1[..., 2] / 2
    b1_y1 = boxes1[..., 1] - boxes1[..., 3] / 2
    b1_x2 = boxes1[..., 0] + boxes1[..., 2] / 2
    b1_y2 = boxes1[..., 1] + boxes1[..., 3] / 2
    
    b2_x1 = boxes2[..., 0] - boxes2[..., 2] / 2
    b2_y1 = boxes2[..., 1] - boxes2[..., 3] / 2
    b2_x2 = boxes2[..., 0] + boxes2[..., 2] / 2
    b2_y2 = boxes2[..., 1] + boxes2[..., 3] / 2
    
    # Get intersection
    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    # Get union
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = box1_area + box2_area - intersection
    
    return intersection / (union + 1e-6)
