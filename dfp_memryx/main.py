import cv2
import numpy as np
from memryx import AsyncAccl
import faiss
import uuid
from datetime import datetime
import sqlite3
from config import get_camera_source, get_camera_location
from face_utils import preprocess_for_arcface, preprocess_for_facenet, preprocess_for_age_gender
from db_utils import init_db, log_to_db, load_embeddings_from_db
import os
import pickle

CAMERA_ID = 0  # Set this to the desired camera id from cameras.json
VIDEO_SOURCE = get_camera_source(CAMERA_ID)
LOCATION = get_camera_location(CAMERA_ID)

dfp_path = "face_age_gender_combined_v3.dfp"
DB_PATH = "face_log.db"

# Initialize the database and create tables if needed
init_db()

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def save_faiss_and_uuids(faiss_index, uuids, faiss_path="faiss.index", uuid_path="uuids.pkl"):
    faiss.write_index(faiss_index, faiss_path)
    with open(uuid_path, "wb") as f:
        pickle.dump(uuids, f)

def load_faiss_and_uuids(faiss_path="faiss.index", uuid_path="uuids.pkl", embedding_dim=512):
    if os.path.exists(faiss_path) and os.path.exists(uuid_path):
        faiss_index = faiss.read_index(faiss_path)
        with open(uuid_path, "rb") as f:
            uuids = pickle.load(f)
        return faiss_index, uuids
    else:
        return faiss.IndexFlatL2(embedding_dim), []

# FAISS setup for embedding search
EMBEDDING_DIM = 128  # Now using 128-dim FaceNet encoding
faiss_index, uuids = load_faiss_and_uuids("faiss.index", "uuids.pkl", EMBEDDING_DIM)
embeddings = []  # List of np.arrays
if len(uuids) == 0:
    # Fallback: rebuild from database
    loaded_uuids, loaded_embeddings = load_embeddings_from_db(DB_PATH)
    if loaded_embeddings:
        for emb, uuid_val in zip(loaded_embeddings, loaded_uuids):
            emb = emb.reshape(1, -1)
            faiss_index.add(emb)
            embeddings.append(emb)
            uuids.append(uuid_val)
    save_faiss_and_uuids(faiss_index, uuids, "faiss.index", "uuids.pkl")
else:
    # If loaded from cache, embeddings are not needed for matching, but can be reconstructed if needed
    pass

# --- Output Handlers (Top Level) ---
FAISS_THRESHOLD = 1.0  # Matching threshold for FAISS (tune as needed)

def make_output_arcface(arcface_embedding):
    def output_arcface(*outs):
        out = outs[0]
        emb = np.squeeze(out)
        print(f"[DEBUG] ArcFace embedding shape: {emb.shape}")
        if emb.shape == (512,):
            arcface_embedding[0] = emb.astype(np.float32)
        else:
            print(f"[WARNING] Invalid ArcFace embedding shape: {emb.shape}, expected (512,)")
    return output_arcface

MAX_AGE = 100  # Used to scale normalized age output from the model
def make_output_age_gender(age_gender_result):
    def output_age_gender(*outs):
        out = outs[0]
        # print("[DEBUG] Full age/gender model output:", out)
        out = np.squeeze(out)
        # print("[DEBUG] Output shape:", out.shape)
        # Case 1: Output is a vector of length >= 103 (e.g., 101 for age, 2 for gender)
        if len(out.shape) == 1 and out.shape[0] >= 103:
            age_probs = out[:101]
            age = float(np.sum(np.arange(101) * age_probs))
            gender_probs = out[101:103]
            gender = float(np.argmax(gender_probs))  # 1=Male, 0=Female
        # Case 2: Output is a vector of length 2 (age, gender_confidence)
        elif len(out.shape) == 1 and out.shape[0] == 2:
            age = float(out[0]) * MAX_AGE  # Scale normalized age
            gender = float(out[1])
        # Case 3: Output is a 2D array (1,2) or (2,1)
        elif out.shape == (1, 2):
            age = float(out[0, 0]) * MAX_AGE
            gender = float(out[0, 1])
        elif out.shape == (2, 1):
            age = float(out[0, 0]) * MAX_AGE
            gender = float(out[1, 0])
        else:
            age, gender = None, None
        age_gender_result[0] = age
        age_gender_result[1] = gender
    return output_age_gender

# --- Main loop ---
def main():
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
        if len(faces) == 0:
            print("No faces detected in the frame.")
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            # Prepare input frames for each model
            arcface_input = preprocess_for_arcface(face_img)      # (112, 112, 3)
            facenet_input = preprocess_for_facenet(face_img)      # (160, 160, 3)
            age_gender_input = preprocess_for_age_gender(face_img) # (224, 224, 3)
            # Flags to ensure each input is only sent once per face
            arcface_called = False
            facenet_called = False
            age_gender_called = False
            gender_called = False
            arcface_embedding = [None]
            facenet_embedding = [None]
            gender_logits_result = [None]
            age_regression_result = [None]
            age_gender_result = [None, None]
            def generate_frame_arcface():
                nonlocal arcface_called
                if arcface_called:
                    return None
                arcface_called = True
                return arcface_input
            def generate_frame_facenet():
                nonlocal facenet_called
                if facenet_called:
                    return None
                facenet_called = True
                return facenet_input
            def generate_frame_age_gender():
                nonlocal age_gender_called
                if age_gender_called:
                    return None
                age_gender_called = True
                return age_gender_input
            def generate_frame_gender():
                nonlocal gender_called
                if gender_called:
                    return None
                gender_called = True
                return age_gender_input  # Same preprocessing as age_gender
            # Output handlers
            output_arcface = make_output_arcface(arcface_embedding)
            output_age_gender = make_output_age_gender(age_gender_result)
            def output_facenet(*outs):
                out = outs[0]
                emb = np.squeeze(out)
                print(f"[DEBUG] FaceNet embedding shape: {emb.shape}")
                facenet_embedding[0] = emb.astype(np.float32) if emb.shape == (128,) else None
            def output_gender_logits(*outs):
                out = outs[0]
                gender_logits_result[0] = np.squeeze(out)
            def output_age_regression(*outs):
                out = outs[0]
                age_regression_result[0] = np.squeeze(out)
            # Connect inputs/outputs for this face
            accl.connect_input(generate_frame_arcface, 0)
            accl.connect_input(generate_frame_facenet, 1)
            accl.connect_input(generate_frame_age_gender, 2)
            accl.connect_input(generate_frame_gender, 3)
            accl.connect_output(output_arcface, 0)
            accl.connect_output(output_facenet, 1)
            accl.connect_output(output_age_gender, 2)
            accl.connect_output(output_gender_logits, 3)
            # Do NOT connect to output 4
            accl.wait()
            # --- Embedding matching and UUID assignment (using FaceNet) ---
            emb = facenet_embedding[0]
            if emb is not None and emb.shape == (128,):
                emb = emb.reshape(1, -1)
                if len(uuids) > 0:
                    D, I = faiss_index.search(emb, 1)
                    print(f"[DEBUG] FAISS distance to closest: {D[0][0]}")
                    if D[0][0] < FAISS_THRESHOLD:
                        matched_uuid = uuids[I[0][0]]
                        print(f"[INFO] Matched existing UUID: {matched_uuid} (distance: {D[0][0]:.4f})")
                    else:
                        matched_uuid = str(uuid.uuid4())
                        faiss_index.add(emb)
                        embeddings.append(emb)
                        uuids.append(matched_uuid)
                        save_faiss_and_uuids(faiss_index, uuids, "faiss.index", "uuids.pkl")
                        print(f"[INFO] New face detected, assigned new UUID: {matched_uuid} (distance: {D[0][0]:.4f})")
                else:
                    matched_uuid = str(uuid.uuid4())
                    faiss_index.add(emb)
                    embeddings.append(emb)
                    uuids.append(matched_uuid)
                    save_faiss_and_uuids(faiss_index, uuids, "faiss.index", "uuids.pkl")
                    print(f"[INFO] First face, assigned new UUID: {matched_uuid}")
            else:
                matched_uuid = "unknown"
                print(f"[WARNING] No valid FaceNet embedding for this face. Assigned UUID: unknown")
            # --- Age from age_gender_mobilenetv2.onnx (output 2) ---
            age = age_gender_result[0]
            # --- Gender from gender_mobilenetv2_v2.onnx (output 3) ---
            gender_logits = gender_logits_result[0]
            if gender_logits is not None:
                exps = np.exp(gender_logits - np.max(gender_logits))
                gender_probs = exps / exps.sum()
                gender_str = "Male" if np.argmax(gender_probs) == 1 else "Female"
            else:
                gender_str = "Unknown"
            # --- Save face image ---
            FACES_DIR = "faces"
            os.makedirs(FACES_DIR, exist_ok=True)
            dt_now = datetime.now()
            face_filename = f"{matched_uuid}_{dt_now.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
            face_path = os.path.join(FACES_DIR, face_filename)
            cv2.imwrite(face_path, face_img)
            # --- Logging ---
            dt = dt_now.isoformat()
            # Only log valid embeddings
            log_to_db(matched_uuid, dt, age, gender_str, CAMERA_ID, LOCATION, face_path, embedding=emb.flatten() if emb is not None and matched_uuid != "unknown" else None)
            print(f"[LOG] UUID: {matched_uuid}, Time: {dt}, Age: {age}, Gender: {gender_str}, Camera: {CAMERA_ID}, Location: {LOCATION}, Image: {face_path}")
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