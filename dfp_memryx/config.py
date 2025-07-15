import json
import os


CAMERA_CONFIG_JSON = os.path.join(os.path.dirname(__file__), 'cameras.json')

with open(CAMERA_CONFIG_JSON, 'r') as f:
    CAMERAS = json.load(f)

# Helper functions

def get_camera_by_id(cam_id):
    """Return the camera dict for the given id, or None if not found."""
    for cam in CAMERAS:
        if cam.get('id') == cam_id:
            return cam
    return None

def get_camera_source(cam_id):
    """Return the source for the given camera id."""
    cam = get_camera_by_id(cam_id)
    return cam['source'] if cam else None

def get_camera_location(cam_id):
    """Return the location for the given camera id."""
    cam = get_camera_by_id(cam_id)
    return cam['location'] if cam else None

def get_camera_name(cam_id):
    """Return the name for the given camera id."""
    cam = get_camera_by_id(cam_id)
    return cam['name'] if cam else None 