from flask import Flask, jsonify, request, render_template, Response
import json
import argparse
import os
import sys
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
import requests

from ultralytics import YOLO
from ultralytics.utils.checks import print_args
from utils.general import update_options

app = Flask(__name__)

# Initialize paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
MODEL_PATH = "./app/KINOKO/yoloresult/okashi23/weights/best.pt"
# Load your YOLO model (adjust the path to your model)
model = YOLO(MODEL_PATH)

def draw_boxes_on_image(image, results):
    """Draw bounding boxes and labels on the image."""
    for result in results:
        name = result["name"]
        confidence = result["confidence"]
        box = result["box"]
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])

        # Define colors for different classes (e.g., kinoko and takenoko)
        color = (0, 255, 0) if name == "kinoko" else (255, 0, 0)  # Green for kinoko, Red for takenoko

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw the label and confidence score
        label = f"{name} {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

def image_to_base64(image):
    """Convert an OpenCV image to a base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        source, save_txt = update_options(request)
        if source is None:
            return jsonify({"error": "No source provided"}), 400

        # Perform prediction with YOLO
        results = model.predict(source=source, save_txt=save_txt, conf=0.25)

        # Extract predictions
        predictions = json.loads(results[0].to_json())  # Use to_json() instead of tojson()

        # Download the image if source is a URL
        if source.startswith('http'):
            response = requests.get(source)
            image = Image.open(BytesIO(response.content))
            image = np.array(image)  # Convert PIL image to numpy array (OpenCV format)
            if image.shape[-1] == 4:  # Handle PNG with alpha channel
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image = cv2.imread(source)
            if image is None:
                return jsonify({"error": "Failed to load image"}), 400

        # Draw bounding boxes on the image
        annotated_image = draw_boxes_on_image(image, predictions)

        # Convert the annotated image to base64
        img_base64 = image_to_base64(annotated_image)

        # Clean up temporary file if it was created
        if request.method == 'POST' and request.files and 'myfile' in request.files:
            temp_path = Path(source)
            if temp_path.exists():
                os.remove(temp_path)

        # Return an HTML page with the annotated image
        return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Prediction Results</title>
        </head>
        <body>
            <h1>Prediction Results</h1>
            <img src="data:image/jpeg;base64,{{ img_base64 }}" alt="Annotated Image">
            <h2>Raw JSON Results</h2>
            <pre>{{ json_results }}</pre>
        </body>
        </html>
        """, img_base64=img_base64, json_results=json.dumps({"results": predictions, "source": source}, indent=2))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=5000, type=int, help='port deployment')
    parser.add_argument('--raw_data', default='raw_data', help='path to raw data')
    opt = parser.parse_args()

    print_args(vars(opt))

    port = opt.port
    delattr(opt, 'port')

    raw_data = Path(opt.raw_data)
    raw_data.mkdir(parents=True, exist_ok=True)
    delattr(opt, 'raw_data')

    app.run(host='0.0.0.0', port=port, debug=False)