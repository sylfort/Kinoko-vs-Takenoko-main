from ultralytics import YOLO
from PIL import Image
import io
import base64

class ImageRecognizer:
    def __init__(self, model_path):
        """
        Initializes the ImageRecognizer with the specified YOLO model.

        Args:
            model_path (str): Path to the YOLO model file (e.g., best.pt).
        """
        self.model = YOLO(model_path)  # Load the model *once*

    def predict_image(self, image_data, image_format="base64"):
        """
        Predicts objects in an image.

        Args:
            image_data (str or bytes or PIL.Image): The image data.  Can be:
                - A base64 encoded string (image_format="base64")
                - Raw bytes of the image (image_format="bytes")
                - A PIL Image object (image_format="pil")
            image_format (str):  Specifies the format of image_data.

        Returns:
            list: A list of dictionaries, where each dictionary represents a
                  detected object.
        """

        if image_format == "base64":
            try:
                image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            except Exception as e:
                raise ValueError(f"Invalid base64 image data: {e}")
        elif image_format == "bytes":
            try:
                image = Image.open(io.BytesIO(image_data))
            except Exception as e:
                raise ValueError(f"Invalid image bytes: {e}")
        elif image_format == "pil":
            image = image_data  # Already a PIL Image object.
        else:
            raise ValueError(f"Invalid image_format: {image_format}")

        results = self.model(image)  # Perform inference

        detections = []
        for result in results:
            for box in result.boxes:
                detection = {
                    'class': int(box.cls[0]),
                    'class_name': result.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'box': {
                        'x1': int(box.xyxy[0][0]),
                        'y1': int(box.xyxy[0][1]),
                        'x2': int(box.xyxy[0][2]),
                        'y2': int(box.xyxy[0][3]),
                    }
                }
                detections.append(detection)

        return detections

    def predict_image_file(self, image_path):
        """
        Predicts objects from an image file path (convenience method).
        """
        image = Image.open(image_path)
        return self.predict_image(image, image_format="pil")