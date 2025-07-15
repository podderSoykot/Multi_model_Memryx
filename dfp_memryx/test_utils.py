import numpy as np
import os
from db_utils import init_db, log_to_db
from face_utils import preprocess_for_arcface, preprocess_for_facenet, preprocess_for_age_gender
import config

print("--- Testing Database Utilities ---")
try:
    init_db()
    print("Database initialized successfully.")
    log_to_db("test-uuid", "2024-01-01T00:00:00", 30, "Male", 0, "TestLocation")
    print("Log entry inserted successfully.")
except Exception as e:
    print(f"Database test failed: {e}")

print("\n--- Testing Face Preprocessing Utilities ---")
try:
    dummy_face = np.ones((256, 256, 3), dtype=np.uint8) * 127  # Gray dummy image
    arcface_img = preprocess_for_arcface(dummy_face)
    facenet_img = preprocess_for_facenet(dummy_face)
    age_gender_img = preprocess_for_age_gender(dummy_face)
    print(f"ArcFace shape: {arcface_img.shape}, dtype: {arcface_img.dtype}")
    print(f"FaceNet shape: {facenet_img.shape}, dtype: {facenet_img.dtype}")
    print(f"Age/Gender shape: {age_gender_img.shape}, dtype: {age_gender_img.dtype}")
except Exception as e:
    print(f"Face preprocessing test failed: {e}")

print("\n--- Testing Camera Config Loading ---")
try:
    print(f"Loaded {len(config.CAMERAS)} cameras from cameras.json")
    for cam in config.CAMERAS:
        print(f"ID: {cam['id']}, Name: {cam['name']}, Location: {cam['location']}, Source: {cam['source']}")
    # Test helper functions
    cam0 = config.get_camera_by_id(0)
    print(f"get_camera_by_id(0): {cam0}")
    print(f"get_camera_source(0): {config.get_camera_source(0)}")
    print(f"get_camera_location(0): {config.get_camera_location(0)}")
    print(f"get_camera_name(0): {config.get_camera_name(0)}")
except Exception as e:
    print(f"Camera config test failed: {e}") 