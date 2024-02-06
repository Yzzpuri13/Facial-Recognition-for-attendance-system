# test_face_recognition.py
import face_recognition

# Load a sample image and test basic functionality
image = face_recognition.load_image_file("yeshu.jpg")
face_locations = face_recognition.face_locations(image)

print("Found {} face(s) in this photograph.".format(len(face_locations)))#check
