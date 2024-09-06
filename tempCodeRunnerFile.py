from flask import Flask, jsonify, request

app = Flask(__name__)

# Store image metadata and alert data globally
image_metadata = {}
alert_data = {}

@app.route('/get_image_metadata', methods=['GET'])
def get_image_metadata():
    return jsonify(image_metadata)

@app.route('/get_alert_data', methods=['GET'])
def get_alert_data():
    return jsonify(alert_data)

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
