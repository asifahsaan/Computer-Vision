import cv2
import face_recognition
import numpy as np
import os

video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.

obama_image = face_recognition.load_image_file("d:/Asif/FaceRecognition/faces/obama.jpeg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

gates_image = face_recognition.load_image_file("d:/Asif/FaceRecognition/faces/bgates.jpeg")
gates_face_encoding = face_recognition.face_encodings(gates_image)[0]

khan_image = face_recognition.load_image_file("d:/Asif/FaceRecognition/faces/khan.jpg")
khan_face_encoding = face_recognition.face_encodings(khan_image)[0]

elon_image = face_recognition.load_image_file("d:/Asif/FaceRecognition/faces/elon.jpg")
elon_face_encoding = face_recognition.face_encodings(elon_image)[0]

hadiqa_image = face_recognition.load_image_file("d:/Asif/FaceRecognition/faces/hadiqa.jpg")
hadiqa_face_encoding = face_recognition.face_encodings(hadiqa_image)[0]

# List of known faces encodings
known_faces = [
    obama_face_encoding,
    gates_face_encoding,
    khan_face_encoding,
    elon_face_encoding,
    hadiqa_face_encoding
]

# List of known faces names
known_names = [
    "Barack Obama",
    "Bill Gates",
    "Imran Khan",
    "Elon Musk",
    "Hadiqa Kiani"
]

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        face_distance = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_names[best_match_index]

        face_names.append(name)

    # Display the name of the recognized person on the frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow("Face Recognition", frame)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
video_capture.release()
cv2.destroyAllWindows()
