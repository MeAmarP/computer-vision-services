# YOLOv1 Training Guide

This document provides guidance on training the YOLOv1 model on the COCO dataset.

## Prerequisites

1. COCO Dataset
   - Download the COCO dataset
   - Update the paths in `config_yolov1.yaml`:
     - `root_dir`: Path to COCO images directory
     - `train_ann_file`: Path to COCO train annotations
     - `val_ann_file`: Path to COCO validation annotations

2. Hardware Requirements
   - GPU with at least 8GB VRAM recommended
   - SSD storage for faster data loading
   - Sufficient RAM (16GB minimum recommended)

## Configuration Setup

1. Model Parameters (`config_yolov1.yaml`)
   - `split_size`: Grid size (default: 7x7)
   - `num_boxes`: Bounding boxes per cell (default: 2)
   - `num_classes`: Number of classes (80 for COCO)

2. Training Parameters
   - Adjust `batch_size` based on available GPU memory
   - Start with default learning rate (`1e-4`)
   - Modify `num_workers` based on CPU cores
   - `lambda_coord` and `lambda_noobj`: Loss weights for coordinate predictions and no-object confidence

## Training Process

1. Initial Validation
   - Start with a small subset of data
   - Verify the training pipeline works
   - Check if loss is decreasing

2. Full Training
   - Run the full training with:
     ```bash
     python train.py
     ```
   - Monitor training progress in `runs/train_[timestamp]/`

3. Training Monitoring
   - Check `training.log` for progress
   - Monitor both training and validation loss
   - Watch for learning rate changes from scheduler
   - Best model saved as `best.pt`
   - Latest checkpoint saved as `last.pt`

## Checkpoints

The training script saves two types of checkpoints:
1. `best.pt`: Model with lowest validation loss
2. `last.pt`: Latest model state

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Training/validation loss
- Current epoch
- Configuration used

## Debugging Tips

1. Dataset Issues
   - Verify image loading
   - Check annotation formatting
   - Validate bounding box coordinates

2. Training Issues
   - Monitor GPU memory usage
   - Check for NaN losses
   - Verify gradient flow
   - Adjust batch size if OOM errors occur

3. Performance Issues
   - Increase `num_workers` for faster data loading
   - Use `pin_memory=True` (already set)
   - Consider using mixed precision training

## Future Enhancements

Planned improvements:
1. mAP (mean Average Precision) calculation
   - Add validation metrics beyond loss
   - Implement COCO evaluation metrics

2. Early Stopping
   - Add patience-based training termination
   - Save computation when model converges

3. Multi-GPU Support
   - Implement DistributedDataParallel
   - Add multi-node training support

4. Visualization
   - Add tensorboard support
   - Implement prediction visualization
   - Add training progress plots

## Known Issues and Solutions

1. Memory Usage
   - If OOM errors occur:
     - Reduce batch size
     - Use gradient accumulation
     - Enable mixed precision training

2. Training Speed
   - Use multiple workers for data loading
   - Enable CUDA synchronization only when needed
   - Use appropriate batch size for GPU

3. Convergence
   - Start with recommended learning rate
   - Monitor validation loss for overfitting
   - Adjust scheduler patience if needed

## Support

For issues and improvements:
1. Check existing issues in the repository
2. Provide training logs when reporting problems
3. Include configuration used when asking for help
