import cv2
import face_recognition
import numpy as np
import csv
import os
import glob
from datetime import datetime

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

#list of known faces encodings
known_faces=[
    obama_face_encoding,
    gates_face_encoding,
    khan_face_encoding,
    elon_face_encoding,
    hadiqa_face_encoding
] 

#list of known faces names
known_names=[
    "Barack Obama",
    "Bill Gates",
    "Imran Khan",
    "Elon Musk",
    "Hadiqa Kiani"
] 

students = known_names.copy()

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

process_this_frame = True

now = datetime.now()
# current_date = now.strftime("%Y/%M/%D")
current_date = now.strftime("%d-%m-%Y")

f = open(current_date+ '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    #rgb_small_frame = small_frame[:, :, ::-1]
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Only process every other frame of video to save time
    if process_this_frame:
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
                print("Checking k naam ata he ya nahi")
                print(name)

            face_names.append(name)
            print("\nChecking naam...")
            print(face_names)

            if name in known_names:
                print("This student is being marked, name is :",name)
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (255,0,0)
                thickness              = 3
                lineType               = 2
 
                cv2.putText(frame,name+' Present', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
                
                if name in students:
                    students.remove(name)
                    print("Baki Students class k")
                    print(students)
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])
                    print("Attendance marked and saved in the file...")

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()