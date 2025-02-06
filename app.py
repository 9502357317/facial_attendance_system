from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("models/face_recognition.h5")

# Database setup
def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, time TEXT)''')
    conn.commit()
    conn.close()

# Recognize face
def recognize_face(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    prediction = model.predict(image)
    return np.argmax(prediction)

# Mark attendance
def mark_attendance(name):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO attendance (name, time) VALUES (?, ?)", (name, current_time))
    conn.commit()
    conn.close()

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Video feed
@app.route('/video_feed')
def video_feed():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Recognize faces and mark attendance
        face_locations = face_recognition.face_locations(frame)
        for (top, right, bottom, left) in face_locations:
            face_image = frame[top:bottom, left:right]
            student_id = recognize_face(face_image)
            student_name = label_names[student_id]
            mark_attendance(student_name)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, student_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
