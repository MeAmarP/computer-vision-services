import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class CocoDataset(Dataset):
    def __init__(self, root_dir, annFile, transform=None, split_size=7, num_boxes=2, num_classes=80):
        """
        COCO Dataset for YOLOv1.
        
        Args:
            root_dir (str): Directory with all the images
            annFile (str): Path to COCO annotation file
            transform (callable, optional): Optional transform to be applied on an image
            split_size (int): Size of grid (SxS)
            num_boxes (int): Number of bounding boxes per grid cell
            num_classes (int): Number of classes
        """
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annFile)
        self.image_ids = list(self.coco.imgs.keys())
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """Get image and target for training."""
        image_id = self.image_ids[idx]
        
        # Load image
        image_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Get image size for normalization
        width = image_info['width']
        height = image_info['height']
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Initialize target tensor
        # Format: [S, S, C + B*5] where:
        # C is number of classes
        # B is number of boxes
        # 5 represents (confidence, x, y, w, h)
        target = torch.zeros(self.split_size, self.split_size, 
                           self.num_classes + self.num_boxes * 5)
        
        # Convert annotations to YOLO format
        for ann in annotations:
            # Get bbox coordinates
            bbox = ann['bbox']  # [x, y, width, height]
            
            # Convert to center format and normalize
            x_center = (bbox[0] + bbox[2]/2) / width
            y_center = (bbox[1] + bbox[3]/2) / height
            bbox_width = bbox[2] / width
            bbox_height = bbox[3] / height
            
            # Get grid cell indices
            grid_x = int(self.split_size * x_center)
            grid_y = int(self.split_size * y_center)
            
            # Ensure indices are within bounds
            grid_x = min(self.split_size - 1, max(0, grid_x))
            grid_y = min(self.split_size - 1, max(0, grid_y))
            
            # Get relative coordinates within grid cell
            x_offset = x_center * self.split_size - grid_x
            y_offset = y_center * self.split_size - grid_y
            
            # Check if we already have a box in this cell
            if target[grid_y, grid_x, self.num_classes] == 0:
                # Set class probability (one-hot encoding)
                category_id = ann['category_id']
                category_idx = self.coco.getCatIds().index(category_id)
                target[grid_y, grid_x, category_idx] = 1
                
                # Set first box coordinates and confidence
                box_idx = self.num_classes
                target[grid_y, grid_x, box_idx] = 1  # confidence
                target[grid_y, grid_x, box_idx+1] = x_offset
                target[grid_y, grid_x, box_idx+2] = y_offset
                target[grid_y, grid_x, box_idx+3] = bbox_width
                target[grid_y, grid_x, box_idx+4] = bbox_height
            
            elif target[grid_y, grid_x, self.num_classes + 5] == 0:
                # Set second box coordinates and confidence
                box_idx = self.num_classes + 5
                target[grid_y, grid_x, box_idx] = 1  # confidence
                target[grid_y, grid_x, box_idx+1] = x_offset
                target[grid_y, grid_x, box_idx+2] = y_offset
                target[grid_y, grid_x, box_idx+3] = bbox_width
                target[grid_y, grid_x, box_idx+4] = bbox_height
        
        return image, target
