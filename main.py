import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

capture_video = cv2.VideoCapture(0)

# Load Known Faces for Attendance
yeshu_image = face_recognition.load_image_file("faces/yeshu.jpg")
yeshu_encoding = face_recognition.face_encodings(yeshu_image)[0]

random_image = face_recognition.load_image_file("faces/random.png")
random_encoding = face_recognition.face_encodings(random_image)[0]

known_face_encodings = [yeshu_encoding, random_encoding]
known_face_names = ["Yeshu", "Random"]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
logged_students = []  # Keep track of logged students

# Open the attendance file
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(f"{current_date}.csv", "w+", newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = capture_video.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Only process every other frame of video to save time
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Input text label with a name below the face
        cv2.rectangle(frame, (left, bottom + 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom + 25), font, 0.5, (255, 255, 255), 1)

        # Log attendance to CSV file
        if name not in logged_students:
            lnwriter.writerow([name, datetime.now().strftime("%H:%M:%S")])
            logged_students.append(name)  # Avoid duplicate logging

    # Display the resulting image
    cv2.imshow('Attendance', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any open OpenCV windows
capture_video.release()
cv2.destroyAllWindows()
f.close()  # Make sure to close the CSV file when done
