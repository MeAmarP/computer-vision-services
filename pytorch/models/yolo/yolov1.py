import torch
import torch.nn as nn

class YOLOv1(nn.Module):
    """
    YOLOv1 model architecture based on the original paper:
    "You Only Look Once: Unified, Real-Time Object Detection"
    
    Args:
        num_classes (int): Number of classes to detect
        num_boxes (int): Number of bounding boxes per cell (default: 2)
        num_cells (tuple): Grid size (S x S) for detection (default: (7, 7))
    """
    def __init__(self, num_classes=20, num_boxes=2, num_cells=(7, 7)):
        super(YOLOv1, self).__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.num_cells = num_cells
        
        # Number of values to predict per box: (x, y, w, h, confidence)
        self.box_values = 5
        
        # Feature extraction (modified GoogLeNet/Inception)
        self.features = nn.Sequential(
            # Initial convolutions
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Reduction layers
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Inception-like blocks (simplified)
            nn.Conv2d(192, 128, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Additional convolutions
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Final convolutions
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )
        
        # Detection head
        self.head = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            
            # Final prediction layer
            nn.Linear(4096, num_cells[0] * num_cells[1] * (num_classes + num_boxes * self.box_values))
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using the approach from the paper"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 448, 448)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, S*S*(num_classes + num_boxes*5))
        """
        x = self.features(x)
        x = self.head(x)
        
        # Reshape output to match grid cell format
        batch_size = x.size(0)
        output_size = self.num_cells[0] * self.num_cells[1] * (self.num_classes + self.num_boxes * self.box_values)
        return x.view(batch_size, self.num_cells[0], self.num_cells[1], -1)
    
    def predict(self, x, conf_threshold=0.5):
        """
        Make predictions with the model
        
        Args:
            x (torch.Tensor): Input tensor
            conf_threshold (float): Confidence threshold for predictions
            
        Returns:
            list: List of predictions (boxes, classes, scores)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            # Process output to get boxes, classes, and scores
            # This will be implemented in the next step
            return output
