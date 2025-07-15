import cv2
import numpy as np

def preprocess_for_arcface(face_img):
    input_img = cv2.resize(face_img, (112, 112))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = input_img.astype(np.float32) / 255.0
    # Output shape: (112, 112, 3)
    return input_img

def preprocess_for_facenet(face_img):
    input_img = cv2.resize(face_img, (160, 160))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = input_img.astype(np.float32) / 255.0
    # Output shape: (160, 160, 3)
    return input_img

def preprocess_for_age_gender(face_img):
    input_img = cv2.resize(face_img, (224, 224))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = input_img.astype(np.float32) / 255.0
    # Output shape: (224, 224, 3)
    return input_img