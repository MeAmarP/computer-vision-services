import tensorflow as tf
import cv2
import numpy as np
from typing import Tuple, Any, Dict


class Inference:
    """
    Handles model loading, prediction, and basic annotation logic
    for images and videos using TensorFlow and OpenCV.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize inference class with config data.
        
        :param config: The dictionary with configuration details.
        """
        self.model_path = config["model"]["model_path"]
        self.input_shape = config["model"].get("input_shape", [224, 224])
        self.task_type = config.get("task_type", "classification")
        
        self.conf_threshold = config["infer_params"].get("confidence_threshold", 0.5)
        self.draw_bboxes = config["infer_params"].get("draw_bboxes", True)
        self.draw_labels = config["infer_params"].get("draw_labels", True)
        
        # Load your TensorFlow model
        # For demonstration, we assume a generic Keras .h5 or SavedModel
        self.model = self._load_model(self.model_path)

    def _load_model(self, path: str) -> tf.keras.Model:
        """
        Loads a TensorFlow model from disk.
        
        :param path: Filesystem path to the model.
        :return: Loaded TensorFlow model.
        """
        print(f"Loading model from {path} ...")
        model = tf.keras.models.load_model(path)
        print("Model loaded successfully.")
        return model

    def predict(self, image: np.ndarray) -> Any:
        """
        Runs inference on a single image (numpy array).
        
        :param image: Input image as a numpy array (BGR format from cv2).
        :return: Model prediction result (format may vary depending on the model).
        """
        # Convert from BGR to RGB if needed
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_image, tuple(self.input_shape[:2]))
        
        # Scale/normalize as needed for your model
        input_data = resized.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)  # shape: (1, H, W, C)
        
        # Inference
        predictions = self.model.predict(input_data)
        return predictions

    def annotate_image(self, image: np.ndarray, predictions: Any) -> np.ndarray:
        """
        Annotates the image with the predictions (e.g., bounding boxes, labels).
        How you annotate depends on the type of task (classification, detection, etc.).
        
        :param image: Original BGR image.
        :param predictions: Inference results from the model.
        :return: Annotated BGR image.
        """
        if self.task_type == "classification":
            # For classification, assume predictions is something like [class_probabilities].
            class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            if confidence > self.conf_threshold:
                label = f"Class: {class_index}, Conf: {confidence:.2f}"
                if self.draw_labels:
                    cv2.putText(
                        image, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
        
        elif self.task_type == "detection":
            # For detection, assume predictions could be boxes + class info
            # Pseudo-code only; real detection models will differ.
            for box in predictions[0]:
                # box = [x1, y1, x2, y2, confidence, class_id]
                x1, y1, x2, y2, conf, cls_id = box
                if conf > self.conf_threshold:
                    if self.draw_bboxes:
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                                      (0, 255, 0), 2)
                    if self.draw_labels:
                        cv2.putText(
                            image, f"ID: {int(cls_id)} {conf:.2f}",
                            (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                        )
        
        # Add other task_types like segmentation or keypoints as needed
        # ...
        
        return image

    def process_image(self, input_path: str, output_path: str) -> None:
        """
        Loads an image from file, performs inference and annotation, then saves the result.
        
        :param input_path: Path to the input image.
        :param output_path: Path to save the annotated image.
        """
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image from {input_path}.")
        
        predictions = self.predict(image)
        annotated_image = self.annotate_image(image, predictions)
        
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved at {output_path}")

    def process_video(self, input_path: str, output_path: str) -> None:
        """
        Loads a video from file, performs inference frame-by-frame, annotates frames,
        and saves the new video to output_path.
        
        :param input_path: Path to the input video file.
        :param output_path: Path to save the annotated video.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video {input_path}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            predictions = self.predict(frame)
            annotated_frame = self.annotate_image(frame, predictions)
            out.write(annotated_frame)
        
        cap.release()
        out.release()
        print(f"Annotated video saved at {output_path}")
