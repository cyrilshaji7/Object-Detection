from flask import Flask, Response, render_template, request, redirect, url_for, send_file, send_from_directory, jsonify
import cv2
from ultralytics import YOLO
from collections import Counter
import datetime
import csv
import io
import threading
import os
import requests
import zipfile
import glob
from pathlib import Path
import time


app = Flask(__name__)

# Load YOLOv8 model

global label_counter, img_counter
# Initialize directories and files
LABELS_FOLDER = 'train/labels'
IMAGE_FOLDER = 'train/images'
COUNTER_FILE = 'label_counter.txt'

# Ensure directories exist
os.makedirs(LABELS_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Ensure label counter file exists and initialize it
if not os.path.isfile(COUNTER_FILE):
    with open(COUNTER_FILE, 'w') as counter_file:
        counter_file.write('1')

# Read the initial label counter
with open(COUNTER_FILE, 'r') as counter_file:
    label_counter = int(counter_file.read().strip())
    img_counter = label_counter
    

# Initialize video capture
cap = cv2.VideoCapture(0)

LABEL_STUDIO_API_URL = 'http://localhost:8080/api/projects/{project_id}/import'
LABEL_STUDIO_API_KEY = '7f6664a6e9a473ea148537b8d367e55d1793b48b'
PROJECT_ID = '3'

# Global variables
all_detections = []
current_counts = Counter()
is_detecting = False
detection_thread = None
image_save_interval = 5  # in seconds
is_training = False
training_lock = threading.Lock() 

label_map = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4, "bus": 5,
    "train": 6, "truck": 7, "boat": 8, "traffic light": 9, "fire hydrant": 10,
    "stop sign": 11, "parking meter": 12, "bench": 13, "bird": 14, "cat": 15,
    "dog": 16, "horse": 17, "sheep": 18, "cow": 19, "elephant": 20, "bear": 21,
    "zebra": 22, "giraffe": 23, "backpack": 24, "umbrella": 25, "handbag": 26,
    "tie": 27, "suitcase": 28, "frisbee": 29, "skis": 30, "snowboard": 31,
    "sports ball": 32, "kite": 33, "baseball bat": 34, "baseball glove": 35,
    "skateboard": 36, "surfboard": 37, "tennis racket": 38, "bottle": 39,
    "wine glass": 40, "cup": 41, "fork": 42, "knife": 43, "spoon": 44, "bowl": 45,
    "banana": 46, "apple": 47, "sandwich": 48, "orange": 49, "broccoli": 50,
    "carrot": 51, "hot dog": 52, "pizza": 53, "donut": 54, "cake": 55,
    "chair": 56, "couch": 57, "potted plant": 58, "bed": 59, "dining table": 60,
    "toilet": 61, "tv": 62, "laptop": 63, "mouse": 64, "remote": 65, "keyboard": 66,
    "cell phone": 67, "microwave": 68, "oven": 69, "toaster": 70, "sink": 71,
    "refrigerator": 72, "book": 73, "clock": 74, "vase": 75, "scissors": 76,
    "teddy bear": 77, "hair drier": 78, "toothbrush": 79
}

runs_dir = 'runs/detect/train/weights'
files = glob.glob(os.path.join(runs_dir, '**/last.pt'), recursive=True)
if not files:
    model = YOLO('yolov8n.pt')
else:
    latest_file = max(files, key=os.path.getmtime)
    model = YOLO(latest_file)
    print(latest_file)
model = YOLO('yolov8n.pt')
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

def detect_objects(frame):
    global all_detections, current_counts
    
    results = model(frame)
    
    current_counts.clear()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            c = box.cls
            class_name = model.names[int(c)]
            current_counts[class_name] += 1
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_detections.insert(0, (dict(current_counts), timestamp))  # Insert at the beginning
    all_detections = all_detections[:100]  # Keep only last 100 entries
    
    annotated_frame = results[0].plot()
    return annotated_frame


@app.route('/label_studio_webhook', methods=['POST'])
def label_studio_webhook():
    
    global img_counter  # Declare label_counter as global
    try:
        data = request.json
        annotation = data.get('annotation')
        if annotation:
            task_id = annotation.get('id')
            label_data = annotation.get('result', [])
            print(label_data[0]['original_width'])
            yolo_labels = []
            for item in label_data:
                original_width = item['original_width']
                original_height = item['original_height']
                value = item['value']

                label = value['rectanglelabels'][0]
                class_number = label_map.get(label, -1)
                print("the class number is: ")
                print(class_number)
                if class_number == -1:
                    continue

                x_center = (value['x'] + value['width'] / 2) / original_width
                y_center = (value['y'] + value['height'] / 2) / original_height
                width = value['width'] / original_width
                height = value['height'] / original_height

                yolo_labels.append(f"{class_number} {x_center} {y_center} {width} {height}\n")
                print(f"{class_number} {x_center} {y_center} {width} {height}\n")
            label_path = os.path.join(LABELS_FOLDER, f"{img_counter}.txt")
            with open(label_path, 'w') as label_file:
                label_file.writelines(yolo_labels)

            # Update label counter and write to file
            img_counter += 1
            
            # with open(COUNTER_FILE, 'w') as counter_file:
            #     counter_file.write(str(label_counter))

            return jsonify({'status': 'success'}), 200
        else:
            return jsonify({'error': 'No annotation data'}), 400

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

import os

@app.route('/retrain')
def retrain():
    global model, is_training
    with training_lock:
        if is_training:
            return jsonify({'status': 'error', 'message': 'Training is already in progress'}), 400
        
        is_training = True

    try:
        # Ensure the data.yaml file exists and is correctly configured
        data_yaml = 'data.yaml'
        if not os.path.isfile(data_yaml):
            return jsonify({'error': 'Training data configuration file not found'}), 400
        
        # Start training
        print("Starting model training...")
        model.train(data=data_yaml, epochs=10, imgsz=640)

        # Return success message
        return jsonify({'status': 'success', 'message': 'Training completed successfully'}), 200

    except Exception as e:
        print(f"Error during training: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

    finally:
        with training_lock:
            is_training = False



def detection_loop():
    global is_detecting, is_training, label_counter
    while is_detecting:
        
        success, frame = cap.read()
        if success:
            detect_objects(frame)
            image_filename = f'{label_counter}.jpg'
            image_path = os.path.join(IMAGE_FOLDER, image_filename)
            cv2.imwrite(image_path, frame)
            
            # Increment the counter and update the file
            label_counter += 1
            with open(COUNTER_FILE, 'w') as counter_file:
                counter_file.write(str(label_counter))

        threading.Event().wait(image_save_interval)

def generate_frames():
    global is_training
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            with training_lock:
                if is_training:
                    #time.sleep(1)  # Pause video feed during training
                    pass
            
            if is_detecting:
                frame = detect_objects(frame)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')


from flask import abort

@app.route('/get_log')
def get_log():


    log_html = "<ul>"
    for counts, timestamp in all_detections:
        log_html += f"<li><span class='timestamp'>{timestamp}</span><br>"
        for obj, count in counts.items():
            log_html += f"<span class='detection'>{obj}: {count}</span><br>"
        log_html += "</li>"
    log_html += "</ul>"
    return log_html

@app.route('/download_csv')
def download_csv():
    output = io.StringIO()
    fieldnames = ['Timestamp'] + list(set(obj for counts, _ in all_detections for obj in counts.keys()))
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for counts, timestamp in all_detections:
        row = {'Timestamp': timestamp}
        row.update({obj: counts.get(obj, 0) for obj in fieldnames[1:]})
        writer.writerow(row)
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='detections.csv'
    )

@app.route('/start_detection')
def start_detection():
    global is_detecting, detection_thread
    with training_lock:
        if not is_detecting:
            if is_training:
                return jsonify({'status': 'error', 'message': 'Cannot start detection while training is in progress'}), 400

            is_detecting = True
            detection_thread = threading.Thread(target=detection_loop)
            detection_thread.start()
            return "Detection started"
        return "Detection already running"

@app.route('/stop_detection')
def stop_detection():
    global is_detecting, detection_thread
    if is_detecting:
        is_detecting = False
        detection_thread.join()
        return "Detection stopped"
    return "Detection is not running"


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = os.path.join(IMAGE_FOLDER, file.filename)
        file.save(filename)
        return redirect(url_for('uploaded_file', filename=file.filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(IMAGE_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True)
