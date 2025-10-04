import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Support both package and script execution contexts
    from .utils import intersection_over_union
except ImportError:  # pragma: no cover
    from utils import intersection_over_union

class YOLOLoss(nn.Module):
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
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction='sum')
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the YOLO loss
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth targets
            
        Returns:
            torch.Tensor: Computed loss
        """
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        targets = targets.reshape_as(predictions)

        pred_classes = predictions[..., :self.C]
        target_classes = targets[..., :self.C]

        pred_boxes = predictions[..., self.C:].view(-1, self.S, self.S, self.B, 5)
        target_boxes = targets[..., self.C:].view(-1, self.S, self.S, self.B, 5)

        target_box = target_boxes[..., 0, :]  # Dataset stores GT in first box slot
        exists_box = target_box[..., 0:1]

        pred_box_xywh = pred_boxes[..., 1:5]
        target_box_xywh = target_box[..., 1:5].unsqueeze(3)

        iou_scores = intersection_over_union(pred_box_xywh, target_box_xywh)
        best_box = iou_scores.argmax(dim=3)
        best_box_one_hot = F.one_hot(best_box, self.B).unsqueeze(-1).float()
        best_pred_boxes = (pred_boxes * best_box_one_hot).sum(dim=3)

        box_predictions = best_pred_boxes[..., 1:5]
        box_targets = target_box[..., 1:5]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        exists_box_box = exists_box.expand_as(box_predictions)
        box_loss = self.mse(
            torch.flatten(exists_box_box * box_predictions, end_dim=-2),
            torch.flatten(exists_box_box * box_targets, end_dim=-2),
        )

        pred_box_conf = best_pred_boxes[..., 0:1]
        target_box_conf = exists_box  # already 1 for cells with objects
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box_conf),
            torch.flatten(exists_box * target_box_conf),
        )

        no_object_mask = 1 - target_boxes[..., 0:1]
        no_object_loss = self.mse(
            torch.flatten(no_object_mask * pred_boxes[..., 0:1]),
            torch.flatten(torch.zeros_like(pred_boxes[..., 0:1])),
        )

        exists_box_class = exists_box.expand_as(pred_classes)
        class_loss = self.mse(
            torch.flatten(exists_box_class * pred_classes, end_dim=-2),
            torch.flatten(exists_box_class * target_classes, end_dim=-2),
        )

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss


# Backwards compatibility alias
class YOLOv1Loss(YOLOLoss):
    pass
