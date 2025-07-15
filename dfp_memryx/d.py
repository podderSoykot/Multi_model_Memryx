import cv2
import numpy as np
from memryx import AsyncAccl
import faiss
import uuid
from datetime import datetime
import sqlite3

# Set your video source: 0 for webcam, or a video file path
VIDEO_SOURCE = 0  # or 'video.mp4'
dfp_path = "face_age_gender_combined.dfp"
DB_PATH = "face_log.db"
CAMERA_INDEX = 0  # For multi-camera, set this per thread
LOCATION = "Way-Wise office"  # For multi-camera, set this per thread

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# FAISS setup for embedding search
EMBEDDING_DIM = 128  # Assuming FaceNet output is 128-dim
faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
embeddings = []  # List of np.arrays
uuids = []       # List of UUID strings

# --- SQLite setup ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS face_log (
        uuid TEXT,
        datetime TEXT,
        age REAL,
        gender TEXT,
        camera INTEGER,
        location TEXT
    )''')
    conn.commit()
    conn.close()

def log_to_db(uuid_val, dt, age, gender, camera, location):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO face_log VALUES (?, ?, ?, ?, ?, ?)", (uuid_val, dt, age, gender, camera, location))
    conn.commit()
    conn.close()

# --- Helper functions for DFP input ---
def preprocess_for_arcface(face_img):
    input_img = cv2.resize(face_img, (112, 112))
    input_img = input_img.astype(np.float32) / 255.0
    input_img = np.expand_dims(input_img, axis=2)  # (112, 112, 1, 3)
    return input_img

def preprocess_for_facenet(face_img):
    input_img = cv2.resize(face_img, (160, 160))
    input_img = input_img.astype(np.float32) / 255.0
    input_img = np.expand_dims(input_img, axis=2)  # (160, 160, 1, 3)
    return input_img

def preprocess_for_age_gender(face_img):
    input_img = cv2.resize(face_img, (224, 224))
    input_img = input_img.astype(np.float32) / 255.0
    input_img = np.expand_dims(input_img, axis=2)  # (224, 224, 1, 3)
    return input_img

# --- Main loop ---
def main():
    init_db()
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("❌ Could not open video source!")
        return
    accl = AsyncAccl(dfp_path)
    window_name = "MemryX DFP Live"
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            # Prepare input frames for each model
            arcface_input = preprocess_for_arcface(face_img)
            facenet_input = preprocess_for_facenet(face_img)
            age_gender_input = preprocess_for_age_gender(face_img)
            # Flags to ensure each input is only sent once per face
            arcface_called = False
            facenet_called = False
            age_gender_called = False
            facenet_embedding = [None]
            age_gender_result = [None, None]
            def generate_frame_arcface():
                nonlocal arcface_called, arcface_input
                if arcface_called:
                    return None
                arcface_called = True
                return arcface_input
            def generate_frame_facenet():
                nonlocal facenet_called, facenet_input
                if facenet_called:
                    return None
                facenet_called = True
                return facenet_input
            def generate_frame_age_gender():
                nonlocal age_gender_called, age_gender_input
                if age_gender_called:
                    return None
                age_gender_called = True
                return age_gender_input
            # Output handlers
            def output_arcface(*outs):
                pass  # Not used for matching in this example
            def output_facenet(*outs):
                out = outs[0]
                emb = np.squeeze(out)
                if emb.shape == (128,):
                    facenet_embedding[0] = emb.astype(np.float32)
            def output_age_gender(*outs):
                out = outs[0]
                out = np.squeeze(out)
                if out.shape == (2,):
                    age = out[0]
                    gender = out[1]
                elif out.shape == (1, 2):
                    age = out[0, 0]
                    gender = out[0, 1]
                elif out.shape == (2, 1):
                    age = out[0, 0]
                    gender = out[1, 0]
                else:
                    age, gender = None, None
                age_gender_result[0] = age
                age_gender_result[1] = gender
            # Connect inputs/outputs for this face
            accl.connect_input(generate_frame_arcface, 0)
            accl.connect_input(generate_frame_facenet, 1)
            accl.connect_input(generate_frame_age_gender, 2)
            accl.connect_output(output_arcface, 0)
            accl.connect_output(output_facenet, 1)
            accl.connect_output(output_age_gender, 2)
            accl.wait()
            # --- Embedding matching and UUID assignment ---
            emb = facenet_embedding[0]
            if emb is not None:
                emb = emb.reshape(1, -1)
                if len(embeddings) > 0:
                    D, I = faiss_index.search(emb, 1)
                    if D[0][0] < 0.8:  # Threshold, tune as needed
                        matched_uuid = uuids[I[0][0]]
                    else:
                        matched_uuid = str(uuid.uuid4())
                        faiss_index.add(emb)
                        embeddings.append(emb)
                        uuids.append(matched_uuid)
                else:
                    matched_uuid = str(uuid.uuid4())
                    faiss_index.add(emb)
                    embeddings.append(emb)
                    uuids.append(matched_uuid)
            else:
                matched_uuid = "unknown"
            # --- Age/Gender ---
            age = age_gender_result[0]
            gender = age_gender_result[1]
            gender_str = "Male" if gender is not None and gender > 0.5 else "Female"
            # --- Logging ---
            dt = datetime.now().isoformat()
            log_to_db(matched_uuid, dt, age, gender_str, CAMERA_INDEX, LOCATION)
            print(f"[LOG] UUID: {matched_uuid}, Time: {dt}, Age: {age}, Gender: {gender_str}, Camera: {CAMERA_INDEX}, Location: {LOCATION}")
            # --- Draw bounding box and info ---
            label = f"{matched_uuid[:8]} | {gender_str}, {age:.1f}" if age is not None else f"{matched_uuid[:8]}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Show the frame
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 