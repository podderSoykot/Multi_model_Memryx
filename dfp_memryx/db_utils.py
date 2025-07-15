import sqlite3
import numpy as np

DB_PATH = "face_log.db"  # You can import or set this as needed

def init_db():
    """
    Initialize the SQLite database and create the face_log table if it does not exist.
    Args:
        db_path (str): Path to the SQLite database file.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS face_log (
        uuid TEXT,
        datetime TEXT,
        age REAL,
        gender TEXT,
        camera INTEGER,
        location TEXT,
        image_path TEXT,
        embedding TEXT  -- New column for face embedding as comma-separated string
    )''')
    conn.commit()
    conn.close()

def log_to_db(uuid_val, dt, age, gender, camera, location, image_path=None, embedding=None):
    """
    Log a face recognition event to the database.
    Args:
        uuid_val (str): UUID of the detected face.
        dt (str): ISO datetime string.
        age (float): Estimated age.
        gender (str): Gender string.
        camera (int): Camera index or ID.
        location (str): Location string.
        image_path (str): Path to the saved face image.
        embedding (np.ndarray or None): Face embedding as a numpy array.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    emb_str = None
    if embedding is not None:
        emb_str = ','.join(map(str, embedding.astype(np.float32).tolist()))
    c.execute("INSERT INTO face_log VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (uuid_val, dt, age, gender, camera, location, image_path, emb_str))
    conn.commit()
    conn.close()

def load_embeddings_from_db(db_path=DB_PATH):
    """
    Load unique UUIDs and their latest embeddings from the database.
    Returns:
        uuids (list of str): List of UUIDs.
        embeddings (list of np.ndarray): List of embeddings.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Get the latest entry for each UUID (by datetime)
    c.execute('''SELECT uuid, embedding FROM face_log WHERE embedding IS NOT NULL AND uuid IN (
        SELECT uuid FROM face_log GROUP BY uuid
    ) ORDER BY datetime DESC''')
    rows = c.fetchall()
    seen = set()
    uuids = []
    embeddings = []
    for uuid_val, emb_str in rows:
        if uuid_val in seen or emb_str is None:
            continue
        emb = np.fromstring(emb_str, sep=',')
        uuids.append(uuid_val)
        embeddings.append(emb)
        seen.add(uuid_val)
    conn.close()
    return uuids, embeddings 