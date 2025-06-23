from PIL import Image
import torch
from pytorch.infer import (
    process_image,
    process_image_segmentation,
    process_image_classification,
    process_image_instance_segmentation,
    process_image_keypoint,
)
from pytorch.utils import generate_color_palette

class DummyDetectionModel:
    def __call__(self, x):
        return [{
            'boxes': torch.tensor([[0, 0, 5, 5]], dtype=torch.float32),
            'labels': torch.tensor([1]),
            'scores': torch.tensor([0.9])
        }]

class DummySegmentationModel:
    def __call__(self, x):
        # Return dictionary with 'out' tensor having shape (1, classes, H, W)
        return {'out': torch.zeros((1, 2, 10, 10))}

class DummyClassificationModel:
    def __call__(self, x):
        return torch.tensor([[0.1, 0.9]])

class DummyInstanceSegmentationModel:
    def __call__(self, x):
        return [{
            'boxes': torch.tensor([[0, 0, 5, 5]], dtype=torch.float32),
            'labels': torch.tensor([0]),
            'scores': torch.tensor([0.9]),
            'masks': torch.zeros((1, 1, 10, 10))
        }]

class DummyKeypointModel:
    def __call__(self, x):
        return [{
            'boxes': torch.tensor([[0, 0, 5, 5]], dtype=torch.float32),
            'labels': torch.tensor([0]),
            'scores': torch.tensor([0.9]),
            'keypoints': torch.tensor([[[2.0, 2.0, 2.0]]])
        }]


def create_test_image(path):
    Image.new('RGB', (10, 10), color='white').save(path)


def test_process_image_detection(tmp_path):
    img_path = tmp_path / 'img.jpg'
    create_test_image(img_path)
    labels = ['obj']
    palette = generate_color_palette(labels)
    out_img = process_image(str(img_path), DummyDetectionModel(), torch.device('cpu'), labels, palette)
    assert isinstance(out_img, Image.Image)


def test_process_image_segmentation(tmp_path):
    img_path = tmp_path / 'img.jpg'
    create_test_image(img_path)
    labels = ['bg', 'fg']
    palette = generate_color_palette(labels)
    out_img = process_image_segmentation(str(img_path), DummySegmentationModel(), torch.device('cpu'), labels, palette)
    assert isinstance(out_img, Image.Image)


def test_process_image_instance_segmentation(tmp_path):
    img_path = tmp_path / 'img.jpg'
    create_test_image(img_path)
    labels = ['obj']
    palette = generate_color_palette(labels)
    out_img = process_image_instance_segmentation(
        str(img_path), DummyInstanceSegmentationModel(), torch.device('cpu'), labels, palette
    )
    assert isinstance(out_img, Image.Image)


def test_process_image_classification(tmp_path):
    img_path = tmp_path / 'img.jpg'
    create_test_image(img_path)
    labels = ['cat', 'dog']
    out_img = process_image_classification(str(img_path), DummyClassificationModel(), torch.device('cpu'), labels)
    assert isinstance(out_img, Image.Image)


def test_process_image_keypoint(tmp_path):
    img_path = tmp_path / 'img.jpg'
    create_test_image(img_path)
    labels = ['person']
    palette = generate_color_palette(labels)
    out_img = process_image_keypoint(
        str(img_path), DummyKeypointModel(), torch.device('cpu'), labels, palette
    )
    assert isinstance(out_img, Image.Image)
