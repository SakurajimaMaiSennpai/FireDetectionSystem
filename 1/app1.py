from flask import Flask, request, jsonify, render_template, Response
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)
model = YOLO('1/best-yolov8.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    file_bytes = np.fromfile(file, np.uint8)
    image_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Perform inference
    results = model(image_np)
    
    # Process results and draw bounding boxes
    boxes = results[0].boxes.xyxy.cpu().numpy()  # xyxy format
    scores = results[0].boxes.conf.cpu().numpy()  # confidence scores
    class_ids = results[0].boxes.cls.cpu().numpy()  # class labels

    # Load class names
    class_names = model.names

    # Draw bounding boxes
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(class_id)]} {score:.2f}"
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Use BGR color for green
        cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Use BGR color for green

    # Convert to JPEG
    _, buffer = cv2.imencode('.jpg', image_np)
    encoded_img_data = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': encoded_img_data})

@app.route('/video_feed')
def video_feed():
    def gen():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform inference
            results = model(frame)
            
            # Get the image with bounding boxes
            annotated_frame = results[0].plot()
            
            # Encode the image as JPEG
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
