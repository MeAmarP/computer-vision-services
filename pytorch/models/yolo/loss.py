import torch
import torch.nn as nn
from utils import intersection_over_union

class YOLOv1Loss(nn.Module):
    """
    YOLOv1 Loss Function implementation as described in the original paper
    
    Args:
        S (int): Grid size (S x S)
        B (int): Number of bounding boxes per cell
        C (int): Number of classes
        lambda_coord (float): Weight for coordinate predictions
        lambda_noobj (float): Weight for no-object predictions
    """
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOv1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the YOLO loss
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth targets
            
        Returns:
            torch.Tensor: Computed loss
        """
        # Reshape predictions to match target shape
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        
        # Get IoU for both predicted boxes with target
        iou_b1 = intersection_over_union(
            predictions[..., 21:25],
            targets[..., 21:25]
        )
        iou_b2 = intersection_over_union(
            predictions[..., 26:30],
            targets[..., 21:25]
        )
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        
        # Get best box and corresponding IoU
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = targets[..., 20].unsqueeze(3)  # Identity of object i in cell j
        
        # ======================== #
        #   FOR BOX COORDINATES   #
        # ======================== #
        
        # Set boxes with no object in them to 0
        box_predictions = exists_box * (
            (
                best_box * predictions[..., 26:30]
                + (1 - best_box) * predictions[..., 21:25]
            )
        )
        
        box_targets = exists_box * targets[..., 21:25]
        
        # Take sqrt of width, height
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )
        
        # ==================== #
        #   FOR OBJECT LOSS   #
        # ==================== #
        
        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        )
        
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * targets[..., 20:21])
        )
        
        # ======================= #
        #   FOR NO OBJECT LOSS   #
        # ======================= #
        
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., 20:21], start_dim=1)
        )
        
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., 20:21], start_dim=1)
        )
        
        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * targets[..., :20], end_dim=-2)
        )
        
        # ================== #
        #   TOTAL LOSS      #
        # ================== #
        
        loss = (
            self.lambda_coord * box_loss  # First two rows of paper
            + object_loss  # Third row of paper
            + self.lambda_noobj * no_object_loss  # Fourth row of paper
            + class_loss  # Fifth row of paper
        )
        
        return loss
