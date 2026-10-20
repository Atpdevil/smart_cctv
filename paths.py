import os

# On serverless (Vercel) only /tmp is writable; everything persistent
# (sqlite db, clips, snapshots, thumbs) must live there.
DATA_DIR = os.environ.get("DATA_DIR") or ("/tmp/data" if os.environ.get("VERCEL") else ".")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "clips"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "thumbs"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "snapshots"), exist_ok=True)


def data_path(*parts):
    return os.path.join(DATA_DIR, *parts)
