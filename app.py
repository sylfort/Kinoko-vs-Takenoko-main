from flask import Flask, render_template, Response, request, jsonify
import json
import argparse
import os
import sys
import cv2
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.checks import print_args
from utils.general import update_options  # Assuming you have utils/general.py from the example

# Initialize paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# --- Configuration ---
MODEL_PATH = "./app/KINOKO/yoloresult/okashi23/weights/best.pt"  # Your custom model path

# Initialize Flask API
app = Flask(__name__)

# Load YOLO Model Globally
model = YOLO(MODEL_PATH)  # Load your custom model here


def predict(opt):
    results = model(**vars(opt), stream=True)

    for result in results:
        if opt.save_txt:
            result_json = json.loads(result.tojson())
            yield json.dumps({'results': result_json, 'source': str(opt.source)})
        else:
            im0 = cv2.imencode('.jpg', result.plot())[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + im0 + b'\r\n')

def predict_single_image(opt):
    # Process the image using your YOLO model
    # This would be similar to your existing predict function but return a single image
    
    # For example:
    model = YOLO(opt.model)
    results = model(opt.source)
    
    # Draw on image
    for r in results:
        im = r.plot()  # Plot the detection results on the image
    
    # Convert the image to bytes
    is_success, buffer = cv2.imencode(".jpg", im)
    return buffer.tobytes()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def video_feed():
    if request.method == 'POST':
        uploaded_file = request.files.get('myfile')
        save_txt = request.form.get('save_txt', 'F')  # Default to 'F' if save_txt is not provided

        if uploaded_file:
            source = Path(__file__).parent / raw_data / uploaded_file.filename
            uploaded_file.save(source)
            opt.source = source
        else:
            opt.source, _ = update_options(request)
            
        opt.save_txt = True if save_txt == 'T' else False
            
    elif request.method == 'GET':
        opt.source, opt.save_txt = update_options(request)

    # Run prediction and get the processed image
    img_bytes = predict_single_image(opt)  # Create a new function that returns an image instead of a stream
    
    # Convert to base64 for embedding in response
    import base64
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return jsonify({'image': f'data:image/jpeg;base64,{img_base64}'})

# @app.route('/predict', methods=['GET', 'POST'])
# def video_feed():
#     if request.method == 'POST':
#         uploaded_file = request.files.get('myfile')
#         save_txt = request.form.get('save_txt', 'F')  # Default to 'F' if save_txt is not provided

#         if uploaded_file:
#             source = Path(__file__).parent / raw_data / uploaded_file.filename # Corrected path
#             uploaded_file.save(source)
#             opt.source = source
#         else:
#             opt.source, _ = update_options(request)

#         opt.save_txt = True if save_txt == 'T' else False

#     elif request.method == 'GET':
#         opt.source, opt.save_txt = update_options(request)

#     return Response(predict(opt), mimetype='multipart/x-mixed-replace; boundary=frame')

# Add this new endpoint to your Flask app
@app.route('/visualize', methods=['POST'])
def visualize():
    data = request.json
    results = data['results']
    source = data['source']
    
    # Download the image
    response = requests.get(source)
    img = Image.open(BytesIO(response.content))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Draw bounding boxes
    for detection in results:
        box = detection['box']
        name = detection['name']
        confidence = detection['confidence']
        
        # Extract coordinates
        x1, y1 = int(box['x1']), int(box['y1'])
        x2, y2 = int(box['x2']), int(box['y2'])
        
        # Choose color based on class
        if name == 'kinoko':
            color = (0, 255, 0)  # Green for kinoko
        else:
            color = (0, 0, 255)  # Red for takenoko
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Add label with confidence
        label = f"{name}: {confidence:.2f}"
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save the image to a temporary file
    output_path = "temp_detection.jpg"
    cv2.imwrite(output_path, img)
    
    # Return the image file
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '--weights', type=str, default=MODEL_PATH, help='model path or triton URL') # Default to your MODEL_PATH
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='source directory for images or videos')
    parser.add_argument('--conf', '--conf-thres', type=float, default=0.25, help='object confidence threshold for detection')
    parser.add_argument('--iou', '--iou-thres', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='image size as scalar or (h, w) list, i.e. (640, 480)')
    parser.add_argument('--half', action='store_true', help='use half precision (FP16)')
    parser.add_argument('--device', default='', help='device to run on, i.e. cuda device=0/1/2/3 or device=cpu')
    parser.add_argument('--show', '--view-img', default=False, action='store_true', help='show results if possible')
    parser.add_argument('--save', action='store_true', help='save images with results')
    parser.add_argument('--save_txt', '--save-txt', action='store_true', help='save results as .txt file')
    parser.add_argument('--save_conf', '--save-conf', action='store_true', help='save results with confidence scores')
    parser.add_argument('--save_crop', '--save-crop', action='store_true', help='save cropped images with results')
    parser.add_argument('--show_labels', '--show-labels', default=True, action='store_true', help='show labels')
    parser.add_argument('--show_conf', '--show-conf', default=True, action='store_true', help='show confidence scores')
    parser.add_argument('--max_det', '--max-det', type=int, default=300, help='maximum number of detections per image')
    parser.add_argument('--vid_stride', '--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--stream_buffer', '--stream-buffer', default=False, action='store_true', help='buffer all streaming frames (True) or return the most recent frame (False)')
    parser.add_argument('--line_width', '--line-thickness', default=None, type=int, help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--visualize', default=False, action='store_true', help='visualize model features')
    parser.add_argument('--augment', default=False, action='store_true', help='apply image augmentation to prediction sources')
    parser.add_argument('--agnostic_nms', '--agnostic-nms', default=False, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--retina_masks', '--retina-masks', default=False, action='store_true', help='whether to plot masks in native resolution')
    parser.add_argument('--classes', type=list, help='filter results by class, i.e. classes=0, or classes=[0,2,3]')  # 'filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--show_boxes', default=True, action='store_false', help='Show boxes in segmentation predictions')
    parser.add_argument('--exist_ok', '--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--raw_data', '--raw-data', default=ROOT / 'data/raw', help='save raw images to data/raw')
    parser.add_argument('--port', default=5000, type=int, help='port deployment')
    opt = parser.parse_args() # Changed to parser.parse_args()

    # print used arguments
    print_args(vars(opt))

    # Get port to deploy
    port = opt.port
    delattr(opt, 'port')

    # Create path for raw data
    raw_data = Path(opt.raw_data)
    raw_data.mkdir(parents=True, exist_ok=True)
    delattr(opt, 'raw_data')


    # Run app
    app.run(host='0.0.0.0', port=port, debug=False) # Don't use debug=True, model will be loaded twice