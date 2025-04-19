import cv2
import pickle
import numpy as np
import os

# Create data directory if it doesn't exist
data_dir = 'data/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Initialize video capture (changed to 0 for default webcam)
video = cv2.VideoCapture(1)
if not video.isOpened():
    raise IOError("Cannot open webcam")

# Load Haar cascade for face detection
facedetect = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

faces_data = []
name = input("Enter your Aadhar number: ")
frames_total = 51
capture_interval = 2  # Capture every 2nd frame
frame_counter = 0      # Tracks frames for capture timing

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    frame_counter += 1  # Increment every frame

    # Capture face every 'capture_interval' frames
    if frame_counter % capture_interval == 0:
        for (x, y, w, h) in faces:
            if len(faces_data) < frames_total:
                # Process and store face
                crop_img = frame[y:y+h, x:x+w]
                resized_img = cv2.resize(crop_img, (50, 50))
                faces_data.append(resized_img)

                # Visual feedback
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
                cv2.putText(frame, f"Captured: {len(faces_data)}/{frames_total}",
                            (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
                break  # Capture only one face per interval

    cv2.imshow('Registration', frame)

    # Exit on 'q' or when enough faces are captured
    if cv2.waitKey(1) == ord('q') or len(faces_data) >= frames_total:
        break

video.release()
cv2.destroyAllWindows()

# Convert and save data
faces_data = np.asarray(faces_data).reshape(len(faces_data), -1)

# Save names
names_path = os.path.join(data_dir, 'names.pkl')
if os.path.exists(names_path):
    with open(names_path, 'rb') as f:
        names = pickle.load(f)
    names += [name] * len(faces_data)
else:
    names = [name] * len(faces_data)

with open(names_path, 'wb') as f:
    pickle.dump(names, f)

# Save face data
faces_path = os.path.join(data_dir, 'faces_data.pkl')
if os.path.exists(faces_path):
    with open(faces_path, 'rb') as f:
        existing_faces = pickle.load(f)
    faces_data = np.vstack((existing_faces, faces_data))

with open(faces_path, 'wb') as f:
    pickle.dump(faces_data, f)

print(f"Registered {len(faces_data)} samples for {name}")
