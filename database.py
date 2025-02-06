import sqlite3
from datetime import datetime

# Initialize the database
def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, time TEXT)''')
    conn.commit()
    conn.close()

# Mark attendance
def mark_attendance(name):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO attendance (name, time) VALUES (?, ?)", (name, current_time))
    conn.commit()
    conn.close()

# Fetch all attendance records
def get_attendance():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT * FROM attendance")
    records = c.fetchall()
    conn.close()
    return records
