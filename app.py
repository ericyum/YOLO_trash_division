

import os
import base64
from flask import Flask, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
from collections import namedtuple

app = Flask(__name__)

# YOLOv8 모델 로드
model = YOLO('C:/Users/SBA/github/trash/best.pt')

# 업로드 폴더 설정
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 허용되는 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# 기록 저장을 위한 데이터 구조
upload_history = []
webcam_history = []
UploadRecord = namedtuple('UploadRecord', ['id', 'filename', 'trash_types', 'is_disposed'])
WebcamRecord = namedtuple('WebcamRecord', ['id', 'frame', 'labels'])

def allowed_file(filename):
    return '.' in filename and            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def gen_frames():  
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        text = "Webcam could not be opened."
        cv2.putText(img, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model(frame, conf=0.3)
            annotated_frame = results[0].plot()

            # 웹캠 기록 저장 (새로운 항목을 맨 앞에 추가)
            if results[0].boxes:
                class_names = results[0].names
                detected_classes_indices = results[0].boxes.cls.tolist()
                labels = [class_names[int(i)] for i in detected_classes_indices]
                if labels:
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    encoded_frame = base64.b64encode(buffer).decode('utf-8')
                    webcam_history.insert(0, WebcamRecord(id=len(webcam_history), frame=encoded_frame, labels=labels))

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            results = model(filepath)
            labels = []
            if results[0].boxes:
                class_names = results[0].names
                detected_classes_indices = results[0].boxes.cls.tolist()
                labels = [class_names[int(i)] for i in detected_classes_indices]

            is_disposed = all(label in ['plastic', 'vinyl', 'Clean Glass Bottle'] for label in labels)
            trash_types = [label for label in labels if 'disposed' not in label]

            # 업로드 기록 저장 (새로운 항목을 맨 앞에 추가)
            upload_history.insert(0, UploadRecord(id=len(upload_history), filename=filename, trash_types=trash_types, is_disposed=is_disposed))

            return render_template('upload.html', filename=filename, trash_types=trash_types, is_disposed=is_disposed)
    return render_template('upload.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/history')
def history():
    RECORDS_PER_PAGE = 6
    upload_page = request.args.get('upload_page', 1, type=int)
    webcam_page = request.args.get('webcam_page', 1, type=int)

    upload_start = (upload_page - 1) * RECORDS_PER_PAGE
    upload_end = upload_start + RECORDS_PER_PAGE
    paginated_upload_history = upload_history[upload_start:upload_end]
    total_upload_pages = (len(upload_history) + RECORDS_PER_PAGE - 1) // RECORDS_PER_PAGE

    webcam_start = (webcam_page - 1) * RECORDS_PER_PAGE
    webcam_end = webcam_start + RECORDS_PER_PAGE
    paginated_webcam_history = webcam_history[webcam_start:webcam_end]
    total_webcam_pages = (len(webcam_history) + RECORDS_PER_PAGE - 1) // RECORDS_PER_PAGE

    return render_template(
        'history.html',
        upload_history=paginated_upload_history,
        webcam_history=paginated_webcam_history,
        upload_page=upload_page,
        total_upload_pages=total_upload_pages,
        webcam_page=webcam_page,
        total_webcam_pages=total_webcam_pages
    )

@app.route('/history_detail/<record_type>/<int:record_id>')
def history_detail(record_type, record_id):
    record = None
    if record_type == 'upload':
        for r in upload_history:
            if r.id == record_id:
                record = r
                break
    elif record_type == 'webcam':
        for r in webcam_history:
            if r.id == record_id:
                record = r
                break
    
    if record:
        return render_template('history_detail.html', record=record, record_type=record_type)
    else:
        return "기록을 찾을 수 없습니다.", 404

if __name__ == '__main__':
    app.run(debug=True)
