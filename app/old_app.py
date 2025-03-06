from flask import Flask, request, jsonify
import sqlite3  
from flask_cors import CORS  # Import the CORS extension
# Import the ImageRecognizer class from the image_recognition module
from image_recognition import ImageRecognizer
import os

# --- Configuration ---
DB_FILE = 'object_detection.db'
MODEL_PATH = "./KINOKO/yoloresult/okashi23/weights/best.pt"  # Update with your model path

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in your app

# --- YOLO Model Loading (using the ImageRecognizer class) ---
recognizer = ImageRecognizer(MODEL_PATH)

# --- Database Helper Functions ---
def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS object_counts (
            object_name TEXT PRIMARY KEY,
            count INTEGER NOT NULL DEFAULT 0
        )
    ''')

    # Check if the records exists.  Use the CORRECT object names.
    cursor.execute("SELECT COUNT(*) FROM object_counts WHERE object_name = 'takenoko'")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO object_counts (object_name) VALUES ('takenoko')")

    cursor.execute("SELECT COUNT(*) FROM object_counts WHERE object_name = 'kinoko'")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO object_counts (object_name) VALUES ('kinoko')")

    conn.commit()
    conn.close()

# --- API Routes ---
@app.route('/')
def index():
     return "Image Recognition API is running!"
     
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.form:
        return jsonify({'error': 'No image provided'}), 400

    image_data = request.form['image']
    if image_data.startswith("data:image"):
        image_data = image_data.split(",")[1]

    try:
        detections = recognizer.predict_image(image_data) # Use the ImageRecognizer

        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                for detection in detections:
                    class_name = detection['class_name']
                    update_query = "UPDATE object_counts SET count = count + 1 WHERE object_name = ?"
                    cursor.execute(update_query, (class_name,))
                conn.commit()
            except sqlite3.Error as e:
                print(f"Error updating database: {e}")
                conn.rollback()
            finally:
                if cursor:  # Check if cursor exists before closing
                    cursor.close()
                conn.close()

        return jsonify(detections)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT object_name, count FROM object_counts")
            results = cursor.fetchall()
            stats = {row['object_name']: row['count'] for row in results}
            return jsonify(stats)
        except sqlite3.Error as e:
            print(f"Error fetching stats: {e}")
            return jsonify({'error': 'Failed to fetch stats'}), 500
        finally:
            cursor.close()
            conn.close()
    else:
      return jsonify({'error': 'Failed to connect to database'}), 500


# --- Run the App ---

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)