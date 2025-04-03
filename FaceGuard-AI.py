import cv2
import face_recognition
import numpy as np
import os
import winsound

# ✅ Set up known faces directory
KNOWN_FACES_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "knownfaces")
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)
    print(f"✅ Folder '{KNOWN_FACES_DIR}' created successfully!")

# ✅ Load known faces
known_face_encodings = []
known_face_names = []

def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    print(f"🔍 Checking images in: {KNOWN_FACES_DIR}")
    files = os.listdir(KNOWN_FACES_DIR)
    print(f"📂 Found files: {files}")

    for filename in files:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            print(f"🔄 Loading {filename}...")
            image = face_recognition.load_image_file(path)

            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
                print(f"✅ Loaded: {filename}")
            else:
                print(f"⚠️ WARNING: No face found in {filename}, skipping it!")

    print(f"🔄 Total known faces loaded: {len(known_face_encodings)}")

# ✅ Load faces at startup
load_known_faces()

# ✅ Optimize Camera Settings
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # ✅ Use DirectShow for better performance
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_counter = 0  # ✅ Frame skipping optimization

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_counter += 1
    frame = cv2.resize(frame, (640, 480))  # ✅ Reduce resolution for faster processing

    if frame_counter % 5 != 0:  # ✅ Only process every 5th frame
        cv2.imshow("Face Recognition Security System", frame)
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # ✅ Super fast face detection
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations[:2])  # ✅ Process only the first 2 faces

    unknown_detected = False

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
            color = (0, 255, 0)  # Green for known persons
            # ❌ Removed beep for known persons
        else:
            unknown_detected = True
            color = (0, 0, 255)  # Red for unknown faces

        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if unknown_detected:
        print("🚨 UNKNOWN PERSON DETECTED!")
        winsound.Beep(1500, 700)

        # ✅ Add Red Tint to the screen
        overlay = frame.copy()
        red_tint = np.zeros_like(frame, dtype=np.uint8)
        red_tint[:, :, 2] = 80  # Adjust Red intensity
        cv2.addWeighted(red_tint, 0.3, frame, 0.7, 0, frame)  # Blend red tint

        cv2.putText(frame, "🚨 WARNING: UNKNOWN PERSON DETECTED!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Face Recognition Security System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and unknown_detected:
        name = input("Enter name for this person: ")
        if name:
            face_image = frame[top:bottom, left:right]
            face_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
            cv2.imwrite(face_path, face_image)
            print(f"✅ Face saved as {face_path}, reloading known faces...")
            load_known_faces()
    
    # ✅ Delete a Saved Person (Press 'D')
    elif key == ord('d'):
        name_to_delete = input("Enter the name of the person to delete: ")
        face_path = os.path.join(KNOWN_FACES_DIR, f"{name_to_delete}.jpg")

        if os.path.exists(face_path):
            os.remove(face_path)  # ✅ Delete the file
            print(f"❌ {name_to_delete} has been deleted!")
            load_known_faces()  # Reload faces after deletion
        else:
            print(f"⚠️ No saved face found with the name '{name_to_delete}'.")

cap.release()
cv2.destroyAllWindows()
