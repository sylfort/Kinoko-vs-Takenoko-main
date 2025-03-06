from bottle import Bottle, request, response, hook
import os
import base64
import io
from PIL import Image
from image_recognition import ImageRecognizer  # Your existing module, modified

# --- Configuration ---
MODEL_PATH = "./KINOKO/yoloresult/okashi23/weights/best.pt"  # Update with your model path

# --- Bottle App Initialization ---
app = Bottle()

# --- CORS Setup (for Bottle) ---
@app.hook('after_request')
def enable_cors():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

@app.route('/', method=['OPTIONS', 'GET'])
def index():
    return "Image Recognition API is running!"

# --- YOLO Model Loading (using the ImageRecognizer class) ---
recognizer = ImageRecognizer(MODEL_PATH)

# --- API Routes ---
@app.route('/predict', method=['OPTIONS', 'POST'])
def predict():
    if request.method == 'OPTIONS':
        return {}

    data = request.json
    if not data or 'image' not in data:
        response.status = 400
        return {'error': 'No image provided'}

    image_data = data['image']
    if image_data.startswith("data:image"):
        image_data = image_data.split(",")[1]

    try:
        # Decode the base64 data.
        image_bytes = base64.b64decode(image_data)

        # Use the ImageRecognizer to get the processed image *as bytes*.
        processed_image_bytes = recognizer.predict_and_draw_boxes(image_bytes)

        # Encode the processed image back to base64 for the response.
        encoded_image = base64.b64encode(processed_image_bytes).decode('utf-8')

        # Return the base64 encoded image.
        return {'processed_image': f"data:image/jpeg;base64,{encoded_image}"}

    except ValueError as e:
        response.status = 400
        return {'error': str(e)}
    except Exception as e:
        response.status = 500
        return {'error': 'An unexpected error occurred'}

# --- Run the App (using Bottle's built-in server for development) ---
if __name__ == '__main__':
    # Bottle's built-in server is fine for *development*.
    # For production, use a production WSGI server like Gunicorn or uWSGI.
    app.run(host='0.0.0.0', port=5000, debug=False, reloader=False)