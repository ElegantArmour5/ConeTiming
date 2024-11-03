from flask import Flask, render_template, request, redirect, url_for
import cv2
import torch

app = Flask(__name__)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
cone_class_id = 0  # Ensure this is the correct class ID for cones in your model

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the video's frames per second
    start_time, end_time = None, None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)
        cones = [det for det in results.xyxy[0] if int(det[5]) == cone_class_id and det[4] > 0.5]  # Confidence threshold

        # Check if any cones are detected
        if cones:
            # Calculate the middle of the detected cone bounding box
            cone_midpoints = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2, conf, cls in cones]
            
            # Check if runner is near cones (you'll need to implement logic for this)
            if start_time is None:
                # Set start time based on first detection of runner near a cone
                start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Convert milliseconds to seconds
            else:
                # Set end time based on detection of runner near the last cone
                end_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    cap.release()
    duration = end_time - start_time if start_time and end_time else 0
    return duration

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = f'uploads/{file.filename}'
        file.save(file_path)
        duration = process_video(file_path)
        return f"Total time taken: {duration:.2f} seconds"

if __name__ == "__main__":
    app.run(debug=True)
