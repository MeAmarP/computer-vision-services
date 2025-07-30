import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

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
    
    return calculate_map_from_tensors(
        predictions, 
        targets,
        config['model']['split_size'],
        config['model']['num_boxes'],
        config['model']['num_classes']
    )

def calculate_map_from_tensors(predictions, targets, S=7, B=2, C=80, iou_threshold=0.5):
    """Calculate mAP from prediction and target tensors."""
    class_predictions = predictions[..., :C]
    box_predictions = predictions[..., C:].view(-1, S, S, B, 5)
    
    class_targets = targets[..., :C]
    box_targets = targets[..., C:].view(-1, S, S, B, 5)
    
    # Calculate AP for each class
    APs = []
    for c in range(C):
        # Get predictions and targets for this class
        class_pred = class_predictions[..., c]
        class_target = class_targets[..., c]
        
        # Calculate precision and recall
        predictions = []
        ground_truths = []
        
        for i in range(S):
            for j in range(S):
                for b in range(B):
                    confidence = box_predictions[..., i, j, b, 0]
                    if confidence > 0:
                        pred_box = box_predictions[..., i, j, b, 1:]
                        predictions.append((confidence, pred_box))
                    
                    target_confidence = box_targets[..., i, j, b, 0]
                    if target_confidence > 0:
                        target_box = box_targets[..., i, j, b, 1:]
                        ground_truths.append(target_box)
        
        if not ground_truths:
            continue
            
        # Sort predictions by confidence
        predictions.sort(key=lambda x: x[0], reverse=True)
        
        TP = torch.zeros(len(predictions))
        FP = torch.zeros(len(predictions))
        total_true_boxes = len(ground_truths)
        
        if total_true_boxes == 0:
            continue
            
        for detection_idx, (confidence, pred_box) in enumerate(predictions):
            best_iou = 0
            
            for idx, target_box in enumerate(ground_truths):
                iou = box_iou(pred_box, target_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou > iou_threshold:
                if confidence > 0.5:
                    TP[detection_idx] = 1
                    ground_truths.pop(best_gt_idx)
            else:
                FP[detection_idx] = 1
                
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        
        recalls = TP_cumsum / (total_true_boxes + 1e-6)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)
        
        # Calculate AP using interpolation
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        
        AP = torch.trapz(precisions, recalls)
        APs.append(AP)
    
    return sum(APs) / len(APs) if APs else 0.0

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
