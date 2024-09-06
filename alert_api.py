from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Store image metadata and alert data globally
image_metadata = {}
alert_data = {}

# Define the directory for uploaded images
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route to serve static images
@app.route('/image/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_image_metadata', methods=['GET'])
def get_image_metadata():
    return jsonify(image_metadata)

@app.route('/get_alert_data', methods=['GET'])
def get_alert_data():
    return jsonify(alert_data)

@app.route('/upload_change_detection_image', methods=['POST'])
def upload_change_detection_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = 'change_detection_image.png'  # Define your file name
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    return jsonify({'message': 'Image uploaded successfully', 'file_path': file_path})

@app.route('/latest-image', methods=['GET'])
def latest_image():
    images_dir = app.config['UPLOAD_FOLDER']
    files = os.listdir(images_dir)
    if not files:
        return jsonify({'error': 'No images found'}), 404
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(images_dir, f)))
    return jsonify(latest_file)

@app.route('/update_image_metadata', methods=['POST'])
def update_image_metadata():
    global image_metadata
    data = request.get_json()
    image_metadata = data
    return jsonify({"message": "Image metadata updated successfully"})

@app.route('/update_alert_data', methods=['POST'])
def update_alert_data():
    global alert_data
    data = request.get_json()
    alert_data = data
    return jsonify({"message": "Alert data updated successfully"})

if __name__ == "__main__":
    app.run(port=5000)
