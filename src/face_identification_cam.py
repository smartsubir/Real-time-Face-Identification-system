import cv2
import face_recognition
import os

def load_known_faces(folder_path):
    known_face_encodings = []
    known_face_names = []

    for person_folder in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person_folder)
        if os.path.isdir(person_path):
            for file_name in os.listdir(person_path):
                if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
                    image_path = os.path.join(person_path, file_name)
                    image = face_recognition.load_image_file(image_path)
                    face_encoding = face_recognition.face_encodings(image)[0]  # Assuming there is only one face in each image
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(person_folder)

    return known_face_encodings, known_face_names

# Load known faces from the 'train_data' folder
train_data_folder = 'train_data'
known_face_encodings, known_face_names = load_known_faces(train_data_folder)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        color = (0, 0, 255)  # Default color for unknown faces (red)

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            color = (0, 255, 0)  # Color for known faces (green)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        font_color = (0, 0, 255) if color == (0, 0, 255) else (0, 255, 0)  # Font color based on frame color
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, font_color, 1)

    cv2.imshow('Face Identification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
