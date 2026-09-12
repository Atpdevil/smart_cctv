---
title: Smart CCTV
emoji: 📹
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
---

# SYNTHETIC SENTINEL — Mission Control

AI-powered multi-camera surveillance: live MJPEG dashboard, YOLOv8 detection,
ByteTrack tracking, cross-camera ReID, browser-drawn restriction zones,
LLM forensic search, clip recording, and object journey timelines.

Status: active development. Backend: Flask + SQLite. Vision: OpenCV /
Ultralytics YOLOv8 / Supervision / TorchReID (OSNet-AIN).

---

## 1. What it does

- **Live monitoring** — MJPEG streams at up to 30 fps with bounding boxes,
  global track IDs, and zone overlays (`/video_feed/<cam_id>`).
- **Detection + tracking** — YOLOv8n (person, car, bike, bus, truck),
  ByteTrack local IDs, OSNet-AIN ReID resolving persistent cross-camera
  global IDs with color-histogram assist (`global_tracker.py`).
- **Zones** — draw restricted zones on the live canvas in the browser;
  intrusions log to SQLite and trigger clip recording (`zone_manager.py`).
- **Threat log** — unified live feed of intrusions, suspicious stays
  (loitering), and person detections (`/api/logs`, `/api/stats`).
- **Intent filter** — natural-language filter via Groq LLM
  (e.g. "show persons only"); RUN/CLR buttons or Enter key
  (`intent_manager.py`, `llm_parser.py`).
- **Forensic search** — natural-language search over the detection index
  with group scoping, thumbnails, clip playback, and timeline view
  (`/api/search`, `/api/timeline`).
- **Camera groups** — logical groupings (corridor, parking, entrance,
  all-feeds) used for filtering and search scope (`camera_groups.json`).
- **Lockdown** — one-click engage/release; engaging snapshots clips on all
  cameras (`/api/lockdown`).
- **Notifications + health** — bell icon polls recent alerts;
  gear icon exposes tracking mode toggle and pipeline health
  (`/api/notifications`, `/api/health`, `/api/mode`).

## 2. Quickstart

Requirements: Python 3.10+, Windows/Linux, ~4 GB RAM (CPU mode).

```bash
git clone https://github.com/vijaykumar-777/smart_cctv.git
cd smart_cctv

python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

Configure environment (`.env`):

```bash
GROQ_API_KEY="gsk_..."
```

Add video sources under `videos/` (`1.mp4`, `2.mp4` map to `cam-01`,
`cam-02` in `app.py` `CAMERAS`), then run:

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

> First boot downloads/loads `yolov8n.pt` and OSNet weights
> (`~/.cache/torch/checkpoints/osnet_ain_x1_0_imagenet.pth`);
> allow ~10 s for pipelines to warm up before the feed appears.

## 3. Using the dashboard

Same dark tactical theme throughout. Every control is functional:

| Area | Control | Action |
|---|---|---|
| Left rail | DASHBOARD / CAM_GRID / LOGS / ZONES / DATA | Jump to and refresh that panel |
| Top bar | TACTICAL / INTEL / FEEDS / CONFIG | Switch context; INTEL focuses search, CONFIG opens settings |
| Top bar | Bell icon | Recent-alerts dropdown (auto-refreshes every 10 s) |
| Top bar | Gear icon | Tracking mode toggle (QUERY/FULL), pipeline health check |
| FEED_SELECT | CAMERA_01 / CAMERA_02 | Switch main MJPEG feed |
| GROUP_FILTER | ALL CAMERAS + groups | Scope forensic search to a camera group |
| ZONE_MANAGEMENT | DRAW_NEW_ZONE | Drag a rectangle on the video, name it, save |
| ZONE list | close icon | Delete zone (with confirm) |
| Video HUD | REC clock, feed label | Live session timer and active feed indicator |
| FORENSIC_SEARCH | SEARCH / CLR | Run LLM-parsed search / clear results |
| Search results | CLIP button | Play recorded clip in modal |
| Search results | TIMELINE button | Open object journey across cameras |
| INTENT_PROCESSOR | RUN / CLR (or Enter) | Apply natural-language class filter / reset to FULL mode |
| THREAT_LOG | auto-refresh 2 s | Live intrusion / stay / detection events |
| INITIATE LOCKDOWN | toggle button | Engage (red, pulsing) / release; triggers clips |
| Mobile bar | INTENT / LIVE / ALERTS / DATA | Same actions, compact layout |
| Modals | CLIP PLAYBACK / TIMELINE | Click backdrop or close icon to dismiss |

Typical flows:

1. **Secure an area** — click DRAW_NEW_ZONE, drag on video, name it.
   Entries trigger THREAT_LOG rows and clip saves under `clips/`.
2. **Filter** — type "show persons only" in INTENT_PROCESSOR, press RUN.
   Press CLR to return to FULL mode.
3. **Investigate** — type "silver car in parking after 6pm" in
   FORENSIC_SEARCH, press SEARCH, open CLIP or TIMELINE from a card.
4. **Emergency** — press INITIATE LOCKDOWN; press again to release.

## 4. Configuration

`app.py` pipeline tuning:

```python
DETECT_EVERY_N   = 1    # YOLO cadence; 3 = 3x faster on CPU
TARGET_FPS       = 30   # MJPEG pacing; 15 = smoother on weak CPU
DETECTION_WIDTH  = 640  # YOLO input width; 480 = faster, 320 = max speed
```

| Goal | Settings |
|---|---|
| Max accuracy (GPU or strong CPU) | `1 / 30 / 640`, JPEG 85 (current defaults) |
| Balanced (laptop CPU) | `3 / 15 / 480`, JPEG 70 |
| Max speed (weak CPU) | `5 / 10 / 320`, JPEG 65 |

Other files: `camera_groups.json` (groups), `videos/` (sources),
`clips/` + `thumbs/` (recordings), `cctv.db` (SQLite), `.env`
(`GROQ_API_KEY` for LLM search/intent; app degrades to empty filters
without it).

## 5. API reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/video_feed/<cam_id>` | MJPEG stream (`cam-01`, `cam-02`) |
| GET | `/api/cameras` | Camera list with feed URLs |
| GET | `/api/stats` | Active objects, intrusions, zones armed |
| GET | `/api/logs` | Recent intrusions / stays / detections |
| GET/POST | `/api/zones` | List / create zone (ratios + name + cam_id) |
| DELETE | `/api/zones/<id>` | Delete zone |
| POST | `/api/intent` | Parse NL filter (`{"query": ...}`) |
| GET/POST | `/api/mode` | Get / set tracking mode (`full`, `query`) |
| GET/POST | `/api/lockdown` | Get / engage-release lockdown |
| GET | `/api/notifications` | Latest 5 alerts + count |
| GET | `/api/health` | Mode, lockdown, per-cam liveness, config |
| GET | `/api/camera-groups` | All camera groups |
| POST | `/api/search` | Forensic search (`query`, optional `group_id`) |
| GET | `/api/timeline?track_id=N` | Object journey across cameras |
| GET | `/clips/<path>`, `/thumbs/<path>`, `/snapshots/<path>` | Media serving |

## 6. Project structure

| Path | Role |
|---|---|
| `app.py` | Flask server, camera workers, MJPEG, all API routes |
| `templates/index.html` | Dashboard UI (single page, vanilla JS) |
| `detector.py` / `tracker.py` | YOLOv8 wrapper / ByteTrack wrapper |
| `reid_model.py` / `global_tracker.py` | OSNet embeddings / cross-cam identity gallery |
| `zone_manager.py` / `event.py` | Zone intrusion logic / entry-stay-exit lifecycle |
| `intent_manager.py` / `llm_parser.py` | NL filter + Groq parsing with fallbacks |
| `query_engine.py` / `search_console.py` | CLI search helpers |
| `clip_recorder.py` / `video_player.py` | Pre/post-event clip writer / event replay |
| `camera_groups.py` + `camera_groups.json` | Group definitions and scope resolution |
| `db.py` / `cctv.db` | Schema, logs, detection index |
| `requirements.txt` / `.env` | Dependencies / secrets |

## 7. Troubleshooting

- **Black feed on first load** — pipelines need ~10 s; hard-reload
  (Ctrl+F5) or re-click the camera button. Confirm
  `/video_feed/cam-01` returns `200 multipart/x-mixed-replace`.
- **Laggy video on CPU** — set `DETECT_EVERY_N=3`, `TARGET_FPS=15`,
  `DETECTION_WIDTH=480` (see table above).
- **Groq 404 model errors** — model names rotate; intent/search fall
  back to empty filters. Update model IDs in `intent_manager.py`
  and `llm_parser.py`.
- **Console `innerHTML of null`** — fixed; every `getElementById` path
  is now guarded. If reintroduced, check new element IDs exist.
- **Tailwind CDN warning** — dev-only; use PostCSS/CLI for production:
  https://tailwindcss.com/docs/installation
- **`message channel closed` error** — browser extension noise, not app code.
- **Windows console encoding crashes** — all `print` calls use ASCII
  tags (`[OK]`, `[Warn]`, `[ReID]`); do not reintroduce raw emoji
  in server logs.

## 8. License

MIT.
