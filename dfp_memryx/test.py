import cv2
import numpy as np
import sys
import os
from memryx import AsyncAccl
from face_utils import preprocess_for_arcface, preprocess_for_facenet, preprocess_for_age_gender

def softmax(logits):
    exps = np.exp(logits - np.max(logits))
    return exps / exps.sum()

# Usage: python test_main_pipeline.py path_to_image.jpg [dfp_file]
if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("Usage: python test_main_pipeline.py path_to_image.jpg [dfp_file]")
    sys.exit(1)

img_path = sys.argv[1]
if not os.path.exists(img_path):
    print(f"Image file not found: {img_path}")
    sys.exit(1)

dfp_path = sys.argv[2] if len(sys.argv) == 3 else "face_age_gender_combined.dfp"
if not os.path.exists(dfp_path):
    print(f"DFP file not found: {dfp_path}")
    sys.exit(1)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread(img_path)
if img is None:
    print(f"Failed to load image: {img_path}")
    sys.exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) == 0:
    print("No faces detected in the image.")
    sys.exit(0)

accl = AsyncAccl(dfp_path)

for i, (x, y, w, h) in enumerate(faces):
    face_img = img[y:y+h, x:x+w]
    arcface_input = preprocess_for_arcface(face_img)
    facenet_input = preprocess_for_facenet(face_img)
    age_gender_input = preprocess_for_age_gender(face_img)
    arcface_embedding = [None]
    facenet_embedding = [None]
    gender_logits_result = [None]
    age_regression_result = [None]
    # Use single-element lists for flags
    arcface_called = [False]
    facenet_called = [False]
    age_gender_called = [False]
    gender_called = [False]
    def generate_frame_arcface():
        if arcface_called[0]:
            return None
        arcface_called[0] = True
        return arcface_input
    def generate_frame_facenet():
        if facenet_called[0]:
            return None
        facenet_called[0] = True
        return facenet_input
    def generate_frame_age_gender():
        if age_gender_called[0]:
            return None
        age_gender_called[0] = True
        return age_gender_input
    def generate_frame_gender():
        if gender_called[0]:
            return None
        gender_called[0] = True
        return age_gender_input
    def output_arcface(*outs):
        out = outs[0]
        arcface_embedding[0] = np.squeeze(out)
    def output_facenet(*outs):
        out = outs[0]
        facenet_embedding[0] = np.squeeze(out)
    def output_gender_logits(*outs):
        out = outs[0]
        gender_logits_result[0] = np.squeeze(out)
    def output_age_regression(*outs):
        out = outs[0]
        age_regression_result[0] = np.squeeze(out)
    accl.connect_input(generate_frame_arcface, 0)
    accl.connect_input(generate_frame_facenet, 1)
    accl.connect_input(generate_frame_age_gender, 2)
    accl.connect_input(generate_frame_gender, 3)
    accl.connect_output(output_arcface, 0)
    accl.connect_output(output_facenet, 1)
    accl.connect_output(output_gender_logits, 3)
    accl.connect_output(output_age_regression, 4)
    accl.wait()
    gender_logits = gender_logits_result[0]
    age_regression = age_regression_result[0]
    gender, age = ("Unknown", None)
    if gender_logits is not None and age_regression is not None:
        gender_probs = softmax(gender_logits)
        gender = "Male" if np.argmax(gender_probs) == 1 else "Female"
        age = int(np.clip(float(age_regression), 0, 1) * 100)
    print(f"Face {i+1}:")
    print(f"  ArcFace embedding shape: {arcface_embedding[0].shape if arcface_embedding[0] is not None else None}")
    print(f"  FaceNet embedding shape: {facenet_embedding[0].shape if facenet_embedding[0] is not None else None}")
    print(f"  Predicted Gender: {gender}")
    print(f"  Predicted Age: {age}")
    out_path = f"test_face_{i+1}.jpg"
    cv2.imwrite(out_path, face_img)
    print(f"  Cropped face saved as: {out_path}") 