import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from tqdm import tqdm
import numpy as np
from PIL import Image
import logging
import sys
import yaml
from datetime import datetime

from yolov1 import YOLOv1
from loss import YOLOLoss
from dataset import CocoDataset
from training_utils import calculate_map, EarlyStopping, VisualizationHelper

logger = logging.getLogger(__name__)

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', leave=False)
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    return total_loss / num_batches

def validate(model, val_loader, criterion, device, config, visualizer=None, epoch=None):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader, desc='Validating', leave=False)):
            images = images.to(device)
            targets = targets.to(device)
            
            predictions = model(images)
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            
            # Visualize predictions periodically
            if visualizer and batch_idx % config['training'].get('viz_interval', 50) == 0:
                visualizer.visualize_batch(images, predictions, targets, epoch, batch_idx)
    
    # Calculate mAP
    mAP = calculate_map(model, val_loader, device, config)
    
    return total_loss / num_batches, mAP

def setup_logging(save_dir):
    """Setup logging configuration."""
    log_file = os.path.join(save_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    # Load consolidated config
    with open('config_yolov1.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(config['logging']['log_dir'], f'train_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    # Setup logging
    setup_logging(save_dir)
    logger.info(f'Config: {config}')

    # Set device
    device = torch.device(f"cuda:{config['hardware'].get('cuda_device', 0)}" if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Initialize model
    model = YOLOv1(
        split_size=config['model']['split_size'],
        num_boxes=config['model']['num_boxes'],
        num_classes=config['model']['num_classes']
    ).to(device)

    # Initialize criterion and optimizer
    criterion = YOLOLoss(
        split_size=config['model']['split_size'],
        num_boxes=config['model']['num_boxes'],
        num_classes=config['model']['num_classes'],
        lambda_coord=config['training']['lambda_coord'],
        lambda_noobj=config['training']['lambda_noobj']
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['lr_scheduler'].get('factor', 0.1),
        patience=config['training']['lr_scheduler'].get('patience', 5),
        min_lr=config['training']['lr_scheduler'].get('min_lr', 1e-6),
        verbose=True
    )

    # Setup transforms
    input_size = config['model'].get('input_size', [448, 448])
    train_transform = transforms.Compose([
        transforms.Resize(tuple(input_size)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(tuple(input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = CocoDataset(
        root_dir=config['data']['root_dir'],
        annFile=config['data']['train_ann_file'],
        transform=train_transform,
        split_size=config['model']['split_size'],
        num_boxes=config['model']['num_boxes'],
        num_classes=config['model']['num_classes']
    )

    val_dataset = CocoDataset(
        root_dir=config['data']['root_dir'],
        annFile=config['data']['val_ann_file'],
        transform=val_transform,
        split_size=config['model']['split_size'],
        num_boxes=config['model']['num_boxes'],
        num_classes=config['model']['num_classes']
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config['training'].get('early_stopping_patience', 7),
        min_delta=config['training'].get('early_stopping_delta', 0),
        verbose=True
    )
    
    # Initialize visualization helper
    visualizer = VisualizationHelper(
        save_dir=os.path.join(save_dir, 'visualizations'),
        class_names=config['data']['class_names']
    )
    os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    best_map = 0.0
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        logger.info(f'Starting epoch {epoch}/{config["training"]["num_epochs"]}')
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, config['training']['num_epochs']
        )
        logger.info(f'Training loss: {train_loss:.4f}')
        
        # Validate
        val_loss, map_score = validate(
            model, val_loader, criterion, device, config,
            visualizer=visualizer, epoch=epoch
        )
        logger.info(f'Validation loss: {val_loss:.4f}, mAP: {map_score:.4f}')
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'mAP': map_score,
            'config': config
        }
        
        # Save latest checkpoint
        torch.save(
            checkpoint,
            os.path.join(save_dir, 'last.pt')
        )
        
        # Save best model by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                checkpoint,
                os.path.join(save_dir, 'best_loss.pt')
            )
            logger.info(f'Saved new best model with validation loss: {val_loss:.4f}')
        
        # Save best model by mAP
        if map_score > best_map:
            best_map = map_score
            torch.save(
                checkpoint,
                os.path.join(save_dir, 'best_map.pt')
            )
            logger.info(f'Saved new best model with mAP: {map_score:.4f}')
        
        # Early stopping check
        if early_stopping(val_loss, model, epoch, os.path.join(save_dir, 'early_stopping.pt')):
            logger.info(f'Early stopping triggered after epoch {epoch}')
            break

if __name__ == '__main__':
    main()
