from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from database import init_db, mark_attendance, get_attendance
from datetime import datetime
from pyngrok import ngrok

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("models/face_recognition.h5")

# Load label names (student names)
label_names = ["Student1", "Student2", "Student3", "Student4", "Student5",
               "Student6", "Student7", "Student8", "Student9", "Student10"]

# Initialize database
init_db()

# Recognize face
def recognize_face(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    prediction = model.predict(image)
    return np.argmax(prediction)

# Video feed generator
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Detect faces
            face_locations = face_recognition.face_locations(frame)
            for (top, right, bottom, left) in face_locations:
                face_image = frame[top:bottom, left:right]
                student_id = recognize_face(face_image)
                student_name = label_names[student_id]
                mark_attendance(student_name)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, student_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Home page
@app.route('/')
def index():
    records = get_attendance()
    return render_template('index.html', records=records)

# Video feed route
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start ngrok tunnel
    public_url = ngrok.connect(5000).public_url
    print(" * Running on", public_url)
    app.run(host='0.0.0.0', port=5000)
