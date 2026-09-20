import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# NEW WAY - Tasks API. No more mp.solutions
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5)

face_landmarker = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
engagement_score = 0

def gen_frames():
    timestamp = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        results = face_landmarker.detect_for_video(mp_image, timestamp)
        timestamp += 1
        
        if results.face_landmarks:
            # Just draw dots for now
            for face_landmarks in results.face_landmarks:
                for landmark in face_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x,y), 1, (0,255,0), -1)
            engagement_score = 80 # dummy for now
        
        cv2.putText(frame, f'Engagement: {engagement_score}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        cv2.imshow('EngageAI Camera', cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR))
        if cv2.waitKey(1) & 0xFF == 27:
            break

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def get_data():
    return jsonify({"engagement": engagement_score})

if __name__ == "__main__":
    print("Starting with Tasks API...")
    app.run(debug=True, host='0.0.0.0', port=5000)