

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
model = YOLO('C:/Users/SBA/github/YOLO_trash_division/result/best.pt')

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
                    is_disposed = all(label in ['plastic', 'vinyl', 'clean_glass_bottle'] for label in labels)
                    if not is_disposed:
                        cv2.putText(annotated_frame, "Incorrectly disposed. See guide below.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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

            is_disposed = all(label in ['plastic', 'vinyl', 'clean_glass_bottle'] for label in labels)
            trash_types = [label for label in labels if 'disposed' not in label]

            # 업로드 기록 저장 (새로운 항목을 맨 앞에 추가)
            upload_history.insert(0, UploadRecord(id=len(upload_history), filename=filename, trash_types=trash_types, is_disposed=is_disposed))

            if not is_disposed:
                guide_html = ''
                if any(label in ['plastic_foreign_substance', 'plastic_labels'] for label in labels):
                    guide_html = '''
                        <img loading="lazy" decoding="async" class="alignnone size-large wp-image-564027" src="//news.seoul.go.kr/env/files/2025/07/6879c90cf316e7.83048194-819x1024.png" alt="250715_기후환경본부_요청_분리배출 가이드라인_v3_4" width="819" height="1024" srcset="https://news.seoul.go.kr/env/files/2025/07/6879c90cf316e7.83048194-819x1024.png 819w, https://news.seoul.go.kr/env/files/2025/07/6879c90cf316e7.83048194-240x300.png 240w, https://news.seoul.go.kr/env/files/2025/07/6879c90cf316e7.83048194-160x200.png 160w, https://news.seoul.go.kr/env/files/2025/07/6879c90cf316e7.83048194-768x960.png 768w, https://news.seoul.go.kr/env/files/2025/07/6879c90cf316e7.83048194-1229x1536.png 1229w, https://news.seoul.go.kr/env/files/2025/07/6879c90cf316e7.83048194-1638x2048.png 1638w, https://news.seoul.go.kr/env/files/2025/07/6879c90cf316e7.83048194-62x78.png 62w, https://news.seoul.go.kr/env/files/2025/07/6879c90cf316e7.83048194-172x215.png 172w" sizes="(max-width: 819px) 100vw, 819px">
                        <td>
                          <p>내용물을 깨끗이 비우고 부착상표(라벨) 등을 제거한 후 가능한 압착하여 뚜껑을 닫아 배출</p>
                          <p>※ 설탕, 유기물 등이 포함된 음료의 경우 물로 헹군 후 배출</p>
                        </td>
                    '''
                elif any(label in ['vinyl_foreign_element'] for label in labels):
                    guide_html = '''
                        <img loading="lazy" decoding="async" class="alignnone size-large wp-image-564028" src="//news.seoul.go.kr/env/files/2025/07/6879c9156cb658.05104728-819x1024.png" alt="250715_기후환경본부_요청_분리배출 가이드라인_v3_5" width="819" height="1024" srcset="https://news.seoul.go.kr/env/files/2025/07/6879c9156cb658.05104728-819x1024.png 819w, https://news.seoul.go.kr/env/files/2025/07/6879c9156cb658.05104728-240x300.png 240w, https://news.seoul.go.kr/env/files/2025/07/6879c9156cb658.05104728-160x200.png 160w, https://news.seoul.go.kr/env/files/2025/07/6879c9156cb658.05104728-768x960.png 768w, https://news.seoul.go.kr/env/files/2025/07/6879c9156cb658.05104728-1229x1536.png 1229w, https://news.seoul.go.kr/env/files/2025/07/6879c9156cb658.05104728-1638x2048.png 1638w, https://news.seoul.go.kr/env/files/2025/07/6879c9156cb658.05104728-62x78.png 62w, https://news.seoul.go.kr/env/files/2025/07/6879c9156cb658.05104728-172x215.png 172w" sizes="(max-width: 819px) 100vw, 819px">
                        <td>
                          <p>내용물을 비우고 물로 헹구는 등 이물질을 제거하여 배출</p>
                          <p>흩날리지 않도록 투명 또는 반투명 봉투에 담아 배출</p>
                          <p>※ 해당품목 예시 : 1회용 봉투 등 각종 비닐류</p>
                          <p>※ 필름·시트형, 랩필름, 각 포장재의 표면적이 50㎤ 미만, 내용물의 용량이 30㎖ 또는 30g이하인 포장재 등 분리배출 표시를 할 수 없는 포장재 포함</p>
                          <p>※ 비해당 품목 : 깨끗하게 이물질 제거가 되지 않은 합필름, 식탁보, 고무장갑, 장판, 돗자리, 섬유류 등(천막, 현수막, 의류, 침구류 등)은 종량제봉투, 특수규격마대 또는 대형폐기물 처리 등 지자체 조례에 따라 배출</p>
                        </td>
                    '''
                elif any(label in ['broken_glass_bottle', 'contaminated_glass_bottle', 'labeled_glass_bottle'] for label in labels):
                    guide_html = '''
                        <img decoding="async" class="alignnone size-large wp-image-564026" src="//news.seoul.go.kr/env/files/2025/07/6879c9047675d1.27797841-819x1024.png" alt="250715_기후환경본부_요청_분리배출 가이드라인_v3_3" width="819" height="1024" srcset="https://news.seoul.go.kr/env/files/2025/07/6879c9047675d1.27797841-819x1024.png 819w, https://news.seoul.go.kr/env/files/2025/07/6879c9047675d1.27797841-240x300.png 240w, https://news.seoul.go.kr/env/files/2025/07/6879c9047675d1.27797841-160x200.png 160w, https://news.seoul.go.kr/env/files/2025/07/6879c9047675d1.27797841-768x960.png 768w, https://news.seoul.go.kr/env/files/2025/07/6879c9047675d1.27797841-1229x1536.png 1229w, https://news.seoul.go.kr/env/files/2025/07/6879c9047675d1.27797841-1638x2048.png 1638w, https://news.seoul.go.kr/env/files/2025/07/6879c9047675d1.27797841-62x78.png 62w, https://news.seoul.go.kr/env/files/2025/07/6879c9047675d1.27797841-172x215.png 172w" sizes="(max-width: 819px) 100vw, 819px">
                        <td>
                          <p>내용물을 비우고 물로 헹구는 등 이물질을 제거하여 배출</p>
                          <p>담배꽁초 등 이물질을 넣지 않고 배출</p>
                          <p>유리병이 깨지지 않도록 주의하여 배출</p>
                          <p>색상별 용기가 설치되어 색상별로 배출이 가능한 경우 분리배출</p>
                          <p>접착제로 부착된 라벨이 아니며 상표제거가 가능한 경우 상표를 제거한 후 배출</p>
                          <p>소주, 맥주 등 빈용기보증금 대상 유리병은 소매점 등으로 반납하면 보증금 환급</p>
                          <p>※ 비해당품목 : 깨진 유리제품(신문지 등에 싸서 종량제 봉투에 배출), 코팅 및 다양한 색상이 들어간 유리제품, 내열 유리제품, 크리스탈 유리제품, 판유리, 조명기구용 유리, 사기·도자기류 등(특수규격마대 또는 대형폐기물 처리 등 지자체 조례에 따라 배출)</p>
                        </td>
                    '''
                return render_template('upload.html', filename=filename, trash_types=trash_types, is_disposed=is_disposed, guide_url='https://news.seoul.go.kr/env/archives/564022', guide_html=guide_html)
            else:
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
