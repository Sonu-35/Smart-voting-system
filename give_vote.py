from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

# Initialize video capture with default camera
video = cv2.VideoCapture(0)
if not video.isOpened():
    raise IOError("Cannot open webcam")

# Correct Haar cascade path
facedetect = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Create data directory if needed
data_dir = 'data/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Load trained data
with open(os.path.join(data_dir, 'names.pkl'), 'rb') as f:
    LABELS = pickle.load(f)

with open(os.path.join(data_dir, 'faces_data.pkl'), 'rb') as f:
    FACES = pickle.load(f)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

COL_NAMES = ['NAME', 'VOTE', 'DATE', 'TIME']

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

        # Face bounding box and text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), 
                   cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    cv2.imshow('Voting System', frame)

    def check_if_exists(value):
        if not os.path.exists("votes.csv"):
            return False
        with open("votes.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == value:
                    return True
        return False

    key = cv2.waitKey(1)
    if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
        try:
            voter_id = output[0]
        except NameError:
            speak("NO FACE DETECTED")
            continue

        if check_if_exists(voter_id):
            speak("YOU HAVE ALREADY VOTED")
            break

        party_map = {
            '1': "BJP",
            '2': "AAP",
            '3': "CONG",
            '4': "NOTA"
        }

        party = party_map[chr(key)]
        speak(f"YOUR VOTE FOR {party} HAS BEEN RECORDED")
        
        # Prepare data entry
        entry = [voter_id, party, date, timestamp]
        
        # Write to CSV
        file_exists = os.path.exists("votes.csv")
        with open("votes.csv", "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(COL_NAMES)
            writer.writerow(entry)
        
        time.sleep(2)
        speak("THANK YOU FOR YOUR PARTICIPATION IN THE ELECTIONS")
        break

    elif key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
