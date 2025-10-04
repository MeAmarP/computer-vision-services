import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import logging

def box_iou(box1, box2):
    """Calculate IoU between box1 and box2."""
    # Get the coordinates of bounding boxes
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    # Get the coordinates of the intersection rectangle
    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both AABBs
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    # Compute the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def _to_float(value):
    if torch.is_tensor(value):
        return float(value.item())
    return float(value)


def _cell_to_corners(grid_y, grid_x, box, grid_size):
    """Convert YOLO (x, y, w, h) relative to cell into absolute corner coords."""
    S = grid_size
    x_offset = _to_float(box[0])
    y_offset = _to_float(box[1])
    w = max(_to_float(box[2]), 1e-6)
    h = max(_to_float(box[3]), 1e-6)

    x_center = (grid_x + x_offset) / S
    y_center = (grid_y + y_offset) / S

    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    return (x1, y1, x2, y2)

def calculate_map(model, val_loader, device, config):
    """Calculate mAP on validation set."""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, batch_targets in tqdm(val_loader, desc='Calculating mAP'):
            images = images.to(device)
            batch_targets = batch_targets.to(device)
            
            batch_predictions = model(images)
            predictions.append(batch_predictions)
            targets.append(batch_targets)
    
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    
    try:
        return calculate_map_from_tensors(
            predictions,
            targets,
            config['model']['split_size'],
            config['model']['num_boxes'],
            config['model']['num_classes'],
        )
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to compute mAP: %s", exc)
        return 0.0

def calculate_map_from_tensors(predictions, targets, S=7, B=2, C=80, iou_threshold=0.5):
    """Calculate mAP from prediction and target tensors."""
    if predictions.numel() == 0:
        return 0.0

    predictions = predictions.view(-1, S, S, C + B * 5)
    targets = targets.view(-1, S, S, C + B * 5)
    batch_size = predictions.shape[0]

    ap_values = []

    for class_idx in range(C):
        detections = []
        ground_truths = {}

        for img_idx in range(batch_size):
            target_cell = targets[img_idx]
            pred_cell = predictions[img_idx]

            for y in range(S):
                for x in range(S):
                    class_target = target_cell[y, x, class_idx]
                    if class_target > 0:
                        gt_box = target_cell[y, x, C + 1:C + 5]
                        ground_truths.setdefault(img_idx, []).append(
                            {'box': _cell_to_corners(y, x, gt_box, S), 'used': False}
                        )

                    for box_idx in range(B):
                        conf = pred_cell[y, x, C + box_idx * 5]
                        conf_value = float(conf.item()) if torch.is_tensor(conf) else float(conf)
                        if conf_value <= 0:
                            continue

                        class_prob = pred_cell[y, x, class_idx]
                        class_prob_value = float(class_prob.item()) if torch.is_tensor(class_prob) else float(class_prob)
                        score = conf_value * class_prob_value
                        if score <= 0:
                            continue

                        box_slice = pred_cell[y, x, C + box_idx * 5 + 1:C + box_idx * 5 + 5]
                        detections.append({
                            'img_idx': img_idx,
                            'score': score,
                            'box': _cell_to_corners(y, x, box_slice, S),
                        })

        if not ground_truths:
            continue

        detections.sort(key=lambda d: d['score'], reverse=True)
        if not detections:
            continue

        tp = torch.zeros(len(detections))
        fp = torch.zeros(len(detections))
        total_true_boxes = sum(len(v) for v in ground_truths.values())

        for det_idx, det in enumerate(detections):
            img_ground_truths = ground_truths.get(det['img_idx'], [])
            if not img_ground_truths:
                fp[det_idx] = 1
                continue

            best_iou = 0.0
            best_gt_idx = -1

            for idx_gt, gt in enumerate(img_ground_truths):
                iou = box_iou(det['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx_gt

            if best_iou >= iou_threshold and best_gt_idx >= 0 and not img_ground_truths[best_gt_idx]['used']:
                tp[det_idx] = 1
                img_ground_truths[best_gt_idx]['used'] = True
            else:
                fp[det_idx] = 1

        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)

        recalls = tp_cumsum / (total_true_boxes + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        precisions = torch.cat((torch.tensor([1.0]), precisions))
        recalls = torch.cat((torch.tensor([0.0]), recalls))

        ap = torch.trapz(precisions, recalls)
        ap_values.append(ap)

    if not ap_values:
        return 0.0

    return float(torch.stack(ap_values).mean().item())

class EarlyStopping:
    """Early stopping handler."""
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss, model, epoch, save_path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch, save_path)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch, save_path)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, val_loss, model, epoch, save_path):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': val_loss,
        }, save_path)

class VisualizationHelper:
    """Helper class for visualization during training."""
    def __init__(self, save_dir, class_names):
        self.save_dir = save_dir
        self.class_names = class_names
        
    def visualize_batch(self, images, predictions, targets, epoch, batch_idx):
        """Visualize a batch of predictions."""
        for i in range(min(4, len(images))):  # Visualize up to 4 images
            image = images[i].cpu()
            pred = predictions[i].cpu()
            target = targets[i].cpu()
            
            # Denormalize image
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
            image = image.permute(1, 2, 0).numpy()
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot original image with ground truth
            ax1.imshow(image)
            self._add_boxes(ax1, target, 'Ground Truth')
            
            # Plot image with predictions
            ax2.imshow(image)
            self._add_boxes(ax2, pred, 'Prediction')
            
            # Save figure
            save_path = f'{self.save_dir}/epoch_{epoch}_batch_{batch_idx}_img_{i}.png'
            plt.savefig(save_path)
            plt.close()
    
    def _add_boxes(self, ax, boxes, title):
        """Add bounding boxes to axis."""
        ax.set_title(title)
        # Convert boxes from YOLO format and add to plot
        # Implementation depends on your box format
        # This is a placeholder for the actual implementation
        pass
