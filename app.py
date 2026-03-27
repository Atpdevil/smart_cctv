import cv2
import os
import threading
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from dotenv import load_dotenv

load_dotenv()
from db import init_db, log_exit, get_stats, get_recent_intrusions, get_zones, query_detection_index
from event import EventManager
from tracker import PersonTracker
from detector import HumanDetector
from zone_manager import ZoneManager
from intent_manager import IntentManager
from mode_manager import ModeManager
from clip_recorder import ClipRecorder
import camera_groups

app = Flask(__name__)

# ── Camera → Video file mapping ──
CAMERAS = {
    "cam-01": {"path": "videos/view-HC3.mp4", "label": "CAM_01_CORRIDOR"},
    "cam-02": {"path": "videos/view-IP2.mp4", "label": "CAM_02_PARKING"},
    "cam-03": {"path": "videos/view-IP5.mp4", "label": "CAM_03_ENTRANCE"},
}

CLASS_NAMES = {
    0: "Person",
    2: "Car",
    3: "Bike",
    5: "Bus",
    7: "Truck"
}

# Initialize DB
init_db()

# Shared singletons
intent_manager = IntentManager()
mode_manager = ModeManager()
clip_recorder = ClipRecorder()
mode_manager.set_mode("query")

# ── Per-camera pipelines ──
cam_pipelines = {}
for cam_id, cam_cfg in CAMERAS.items():
    pipeline = {
        "detector": HumanDetector(),
        "tracker": PersonTracker(),
        "event_manager": EventManager(camera_id=cam_id),
        "zone_manager": ZoneManager(camera_id=cam_id),
        "video_path": cam_cfg["path"],
        "label": cam_cfg["label"],
    }
    # Register suspicious callback per camera
    def _make_callback(cid):
        def _on_suspicious(track_id, duration):
            clip_recorder.trigger_clip(cid, "suspicious_stay", track_id, None)
        return _on_suspicious
    pipeline["event_manager"].on_suspicious(_make_callback(cam_id))
    cam_pipelines[cam_id] = pipeline


def generate_frames(cam_id):
    """MJPEG generator for a specific camera."""
    pipeline = cam_pipelines.get(cam_id)
    if not pipeline:
        return

    cap = cv2.VideoCapture(pipeline["video_path"])
    if not cap.isOpened():
        print(f"Error: Could not open video for {cam_id}: {pipeline['video_path']}")
        return

    detector = pipeline["detector"]
    tracker = pipeline["tracker"]
    event_mgr = pipeline["event_manager"]
    zone_mgr = pipeline["zone_manager"]

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        detections = detector.detect(frame)
        tracked_objects = tracker.update(frame, detections)
        event_mgr.update(tracked_objects, frame)
        zone_mgr.draw_zones(frame)

        for (x1, y1, x2, y2, track_id, cls_id) in tracked_objects:
            process_object = True

            if mode_manager.is_query_mode():
                process_object = intent_manager.match(cls_id)
            if mode_manager.is_query_mode() and not process_object:
                continue

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            label = CLASS_NAMES.get(cls_id, "Object")

            if process_object and cls_id == 0:
                intrusion_result = zone_mgr.check_intrusion(
                    track_id, center_x, center_y, pipeline["video_path"], video_time
                )
                if intrusion_result.get("triggered"):
                    clip_recorder.trigger_clip(
                        cam_id, "intrusion", track_id, intrusion_result.get("zone_name")
                    )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)

        clip_recorder.push_frame(cam_id, frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
@app.route('/video_feed/<cam_id>')
def video_feed(cam_id="cam-01"):
    if cam_id not in CAMERAS:
        cam_id = "cam-01"
    return Response(generate_frames(cam_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/cameras')
def api_cameras():
    result = []
    for cam_id, cfg in CAMERAS.items():
        result.append({
            "cam_id": cam_id,
            "label": cfg["label"],
            "feed_url": f"/video_feed/{cam_id}"
        })
    return jsonify(result)


@app.route('/api/stats')
def api_stats():
    return jsonify(get_stats())

@app.route('/api/logs')
def api_logs():
    return jsonify(get_recent_intrusions(limit=10))

@app.route('/api/zones', methods=['GET'])
def api_get_zones():
    return jsonify(get_zones())

@app.route('/api/zones', methods=['POST'])
def api_post_zones():
    data = request.json
    name = data.get('name', 'Custom Zone')
    cam_id = data.get('cam_id', 'cam-01')
    x1_ratio = float(data.get('x1_ratio', 0))
    y1_ratio = float(data.get('y1_ratio', 0))
    x2_ratio = float(data.get('x2_ratio', 0))
    y2_ratio = float(data.get('y2_ratio', 0))

    pipeline = cam_pipelines.get(cam_id)
    if not pipeline:
        return jsonify({"error": "Unknown cam_id"}), 400

    cap = cv2.VideoCapture(pipeline["video_path"])
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    x1 = int(x1_ratio * width)
    y1 = int(y1_ratio * height)
    x2 = int(x2_ratio * width)
    y2 = int(y2_ratio * height)

    pipeline["zone_manager"].save_zone(name, x1, y1, x2, y2, "restricted")
    return jsonify({"status": "success"})

@app.route('/api/zones/<int:zone_id>', methods=['DELETE'])
def api_delete_zones(zone_id):
    # Delete from all zone managers
    for p in cam_pipelines.values():
        p["zone_manager"].delete_zone(zone_id)
    return jsonify({"status": "success"})

@app.route('/api/intent', methods=['POST'])
def api_intent():
    data = request.json
    query = data.get('query', '')
    intent_manager.set_intent(query)
    mode_manager.set_mode("query")
    return jsonify(intent_manager.intent)

# ── Camera Groups ──
@app.route('/api/camera-groups')
def api_camera_groups():
    return jsonify(camera_groups.get_all_groups())

# ── Forensic Search ──
@app.route('/api/search', methods=['POST'])
def api_search():
    try:
        data = request.json or {}
        query = data.get('query', '')
        group_id = data.get('group_id')

        if not query:
            return jsonify({"error": "query is required"}), 400

        filters = intent_manager.parse_search_query(query)

        if group_id:
            cam_ids = camera_groups.get_cameras_for_group(group_id)
            if cam_ids:
                filters["cam_ids"] = cam_ids

        results = query_detection_index(filters)

        for r in results:
            if r.get("clip_path"):
                r["clip_url"] = "/" + r["clip_path"].replace("\\", "/")
            else:
                r["clip_url"] = None
            if r.get("thumb_path"):
                r["thumb_url"] = "/" + r["thumb_path"].replace("\\", "/")
            else:
                r["thumb_url"] = None

            grp = camera_groups.get_group_for_camera(r.get("cam_id", ""))
            r["group_label"] = grp["label"] if grp else "Unknown"

        cameras_searched = list(set(r.get("cam_id", "") for r in results))

        return jsonify({
            "query": query,
            "parsed_filters": filters,
            "cameras_searched": cameras_searched,
            "result_count": len(results),
            "results": results
        })
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({"error": str(e)}), 500

# ── Timeline ──
@app.route('/api/timeline')
def api_timeline():
    track_id = request.args.get('track_id')
    if not track_id:
        return jsonify({"error": "track_id is required"}), 400
    try:
        results = query_detection_index({"track_id": int(track_id)})
        results.reverse()
        for r in results:
            if r.get("clip_path"):
                r["clip_url"] = "/" + r["clip_path"].replace("\\", "/")
            else:
                r["clip_url"] = None
            grp = camera_groups.get_group_for_camera(r.get("cam_id", ""))
            r["group_label"] = grp["label"] if grp else "Unknown"
        return jsonify({"track_id": int(track_id), "timeline": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Static file serving ──
@app.route('/clips/<path:filepath>')
def serve_clip(filepath):
    clips_dir = os.path.join(os.path.dirname(__file__), 'clips')
    try:
        return send_from_directory(clips_dir, filepath)
    except Exception:
        return jsonify({"error": "Clip not found"}), 404

@app.route('/thumbs/<path:filepath>')
def serve_thumb(filepath):
    thumbs_dir = os.path.join(os.path.dirname(__file__), 'thumbs')
    try:
        return send_from_directory(thumbs_dir, filepath)
    except Exception:
        return jsonify({"error": "Thumbnail not found"}), 404

@app.route('/snapshots/<path:filepath>')
def serve_snapshot(filepath):
    snap_dir = os.path.join(os.path.dirname(__file__), 'snapshots')
    try:
        return send_from_directory(snap_dir, filepath)
    except Exception:
        return jsonify({"error": "Snapshot not found"}), 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)