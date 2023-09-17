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

known_people = {
    "Barack Obama": {
        "height": "6'1\"",
        "age": 60,
        "country": "USA",
        "designation": "Former President"
    },
    "Bill Gates": {
        "height": "5'10\"",
        "age": 66,
        "country": "USA",
        "designation": "Philanthropist"
    },
    "Imran Khan": {
        "height": "6'0\"",
        "age": 69,
        "country": "Pakistan",
        "designation": "Prime Minister"
    },
    "Elon Musk": {
        "height": "6'2\"",
        "age": 50,
        "country": "USA",
        "designation": "Entrepreneur"
    },
    "Hadiqa Kiani": {
        "height": "5'6\"",
        "age": 47,
        "country": "Pakistan",
        "designation": "Singer"
    }
}

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    face_details = []

    if len(face_locations) == 0:
        # No face detected, display a waiting message
        cv2.putText(frame, "Waiting for face...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"
            details = {}

            face_distance = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                name = known_names[best_match_index]
                # Get the additional details for the recognized person
                details = known_people.get(name, {})

            face_names.append(name)
            face_details.append(details)

        detail_frame = np.zeros_like(frame)
        for (top, right, bottom, left), name, details in zip(face_locations, face_names, face_details):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a red box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            
            # Combine the person's name and details into one string
            # detail_text = f"Name: {name} Height: {details.get('height', '')} Age: {details.get('age', '')} Country: {details.get('country', '')} Designation: {details.get('designation', '')}"
            # Combine the person's name and details into multiple lines
            detail_text = f"Name: {name}\n"
            detail_text += f"Height: {details.get('height', '')}\n"
            detail_text += f"Age: {details.get('age', '')}\n"
            detail_text += f"Country: {details.get('country', '')}\n"
            detail_text += f"Designation: {details.get('designation', '')}"
            
            # Draw the combined text on the frame
            # Split the text into lines and draw each line on the frame
            lines = detail_text.split('\n')
            line_height = 20  # Adjust this value to control the spacing between lines
            text_y = bottom + 20  # Adjust this value to control the vertical position of the text
            for line in lines:
                text_size = cv2.getTextSize(line, font, 0.5, 1)[0]
                cv2.rectangle(frame, (left + 4, text_y - text_size[1] - 2), (left + text_size[0] + 6, text_y + 2), (0, 0, 0), cv2.FILLED)  # Background color
                cv2.putText(frame, line, (left + 6, text_y), font, 0.5, (255, 255, 255), 1)
                text_y += line_height
            # cv2.putText(frame, detail_text, (left + 6, top - 6), font, 0.5, (255, 255, 255), 1)
            # cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            # detail_text = f"Height: {details.get('height', '')} Age: {details.get('age', '')} Country: {details.get('country', '')} Designation: {details.get('designation', '')}"
            # cv2.putText(detail_frame, detail_text, (10, 30), font, 0.5, (255, 255, 255), 1)

        cv2.imshow("Face Recognition", frame)
        # cv2.imshow("Details", detail_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()

            
            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_names[first_match_index]
            #     students.remove(name)
            #     print(students)
            #     if len(students) == 0:
            #         print('All Students are Present')
            #         students = known_names.copy()
            #         print(students)
            #         with open('d:/Asif/FaceRecognition/attendance.csv', 'a') as csvFile:
            #             writer = csv.writer(csvFile)
            #             writer.writerow([datetime.now()])
            #         csvFile.close()
            #         print('Attendance Marked')
            #         print('Attendance Marked')


# known_faces_dir = 'known_faces'
# for filename in glob.glob(os.path.join(known_faces_dir, '*.jpg')):
#     name = os.path.splitext(os.path.basename(filename))[0]
#     known_names.append(name)
#     image = face_recognition.load_image_file(filename)
#     face_encoding = face_recognition.face_encodings(image)[0]
#     known_faces.append(face_encoding)
#     print('Known Faces: ', len(known_faces))
#     print('Total Known Face Encodings:',len(known_faces),'\n')
#     print('\n\n')
#     print('Done Loading Known Faces.')
#     print('End Loop.\n')
#     print('Known Names: ', known_names)
#     print('Total Known Names:',len(known_names),'\n')
#     print('\n\n')