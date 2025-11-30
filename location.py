# <--- entire file: paste this to replace your current app file --->
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import csv
import hashlib
import secrets
import re
import threading
import json
from collections import deque
from contextlib import closing
from datetime import datetime, timedelta
import numpy as np
import streamlit as st
import base64

import pandas as pd
import altair as alt

try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
except Exception:
    mp = None

try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    from av import VideoFrame
except Exception:
    webrtc_streamer = None
    VideoTransformerBase = object
    VideoFrame = None

audio_enabled = False
try:
    import pygame
    from gtts import gTTS

    pygame.mixer.init()
    audio_enabled = True
except Exception:
    audio_enabled = False

# Optional Lottie
try:
    from streamlit_lottie import st_lottie
    import requests
except Exception:
    st_lottie = None
    requests = None

st.set_page_config(page_title="Next Gen Fitness Tracker", layout="wide", page_icon="🏋")
os.makedirs("sounds", exist_ok=True)
os.makedirs("user_logs", exist_ok=True)
DB_PATH = "users.db"

# ------------ NEW: Calories per rep (used live + saved) ------------
CALORIES_PER_REP = {
    "squat": 0.32,
    "pushup": 0.29,
    "jumping_jack": 0.20,
    "high_knees": 0.18,
    "arm_raise": 0.10,
}

# -------------------- CSS / Styling --------------------
def inject_css():
    st.markdown(
        """
        <style>
        /* Display Serif for brand/hero titles */
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800;900&display=swap');
        :root{
          --brand-red:#ef4444;
          --brand-bg:#0b0f17;
          --card:#0f172a;
          --card-2:#111827;
          --text:#e5e7eb;
          --muted:#94a3b8;
          --ring: rgba(239,68,68,.25);
        }

        html, body, .block-container{background:var(--brand-bg)!important; color:var(--text)!important}
        /* Keep comfortable padding for regular pages (Login/Register) */
        .block-container{padding-top:3rem!important; padding-bottom:1rem!important; overflow:visible!important; border-top:none!important; box-shadow:none!important}
        /* Ensure app view container has enough headroom and no clipping */
        [data-testid="stAppViewContainer"]{padding-top:3rem!important; overflow:visible!important; border-top:none!important}
        /* Remove fixed Streamlit header to prevent overlap/cropping */
        [data-testid="stHeader"]{display:none!important}
        [data-testid="stDecoration"]{display:none!important}
        [data-testid="stToolbar"]{display:none!important}
        .stApp{overflow:visible!important}
        .stMarkdown, .topbar, .brand-title{overflow:visible!important}

        section[data-testid="stSidebar"] * {font-size:16px!important}

        /* Topbar layout to prevent title clipping */
        .topbar{display:flex; align-items:flex-start; gap:16px; width:100%; flex-wrap:wrap; padding-top:10px; margin-top:0}
        .topbar .brand-title{flex:1 1 auto; min-width:280px}
        .topbar .user-pill{flex:0 0 auto}
        @media (max-width: 640px){
          .topbar{gap:10px}
        }

        .brand-title{
          font-family: 'Playfair Display', ui-serif, Georgia, 'Times New Roman', serif;
          font-weight:800; font-size:clamp(28px, 6vw, 46px); letter-spacing:.002em; margin:4px 0 14px 0; padding-top:10px;
          position:relative; display:block; z-index:2; line-height:1.25; text-rendering:optimizeLegibility;
          max-width:100%; white-space:normal; word-break:normal; overflow-wrap:break-word; overflow:visible; hyphens:auto;
        }
        .brand-title .next{color:#ffffff; font-weight:800;}
        .brand-title .gen{color:var(--brand-red); font-weight:900;}

        .pill { font-size:13px; padding:6px 12px; border-radius:9999px; background:#111827; color:#e5e7eb; border:1px solid rgba(255,255,255,0.06); }

        .grid{display:grid; grid-template-columns:repeat(12,minmax(0,1fr)); gap:16px}
        .col-3{grid-column:span 3 / span 3}
        .col-4{grid-column:span 4 / span 4}
        .col-6{grid-column:span 6 / span 6}
        .col-12{grid-column:span 12 / span 12}
        .card{background:var(--card); border:1px solid rgba(255,255,255,0.06); border-radius:14px; padding:16px}
        .soft{background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01))}
        .title-xs{font-size:13px; color:var(--muted); font-weight:700; text-transform:uppercase; letter-spacing:.06em}
        .value-xl{font-size:34px; font-weight:900; line-height:1; margin-top:6px}
        .subtext{font-size:12px; color:var(--muted)}

        .achv{display:flex; align-items:center; gap:12px; padding:12px; border-radius:12px; border:1px solid rgba(255,255,255,0.06); background:var(--card-2)}
        .achv .left{font-size:22px}
        .achv .meta{flex:1}
        .achv .meta .name{font-weight:800}
        .bar{height:8px; background:#1f2937; border-radius:9999px; overflow:hidden; margin-top:6px; border:1px solid rgba(255,255,255,0.06)}
        .bar > span{display:block; height:100%; width:0}
        .bar.red > span{background:var(--brand-red)}

        section[data-testid="stSidebar"] input[type="radio"]{ accent-color: var(--brand-red); }
        section[data-testid="stSidebar"] .stRadio svg circle{ fill: var(--brand-red)!important; }

        .hint { color: #cbd5e1; font-size:14px; margin-top:6px }

        /* Camera holder */
        .camera-shell{background:linear-gradient(135deg,#0f172a,#1e293b); border-radius:12px; overflow:hidden; box-shadow:0 0 20px rgba(0,0,0,0.4)}
        .camera-title{background:#111827; color:#10b981; display:flex; justify-content:space-between; padding:6px 12px; font-size:14px}
        .pill-rec{color:#f87171; font-weight:bold}
        /* Dark info tiles (so text is visible) */
        .tile-dark{background:#0e1726; border:1px solid #1f2937; padding:12px; border-radius:10px; text-align:center}
        .tile-dark .big{font-size:24px; font-weight:900; color:#ffffff}
        .tile-dark .label{font-size:12px; color:#93a4b8}

        /* Dashboard panels and pills */
        .panel{background:#0e1623; border:1px solid rgba(255,255,255,0.06); border-radius:14px; padding:16px}
        .pill-row{display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px}
        .pill-card{background:#0f172a; border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:12px; display:flex; align-items:center; gap:10px}
        .pill-ic{width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; background:#1f2937; font-size:18px}
        .pill-meta .ttl{font-weight:800; font-size:13px}
        .pill-meta .sub{color:#9ca3af; font-size:12px}

        /* Workout tiles */
        .tiles{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px}
        .w-tile{background:#0f172a; border:1px solid rgba(255,255,255,0.06); border-radius:12px; overflow:hidden}
        .w-img{width:100%; height:120px; object-fit:cover; filter:saturate(1.1)}
        .w-cap{padding:10px; font-weight:700}

        /* Donut progress */
        .donut-wrap{display:flex; align-items:center; gap:16px}
        .donut-box{position:relative; width:120px; height:120px}
        .donut{--p:65; width:120px; height:120px; border-radius:50%; background:
          conic-gradient(#22d3ee calc(var(--p)*1%), #1f2937 0); display:grid; place-items:center;}
        .donut::after{content:""; width:84px; height:84px; background:#0e1623; border-radius:50%; box-shadow:inset 0 0 0 1px rgba(255,255,255,.06)}
        .donut-val{position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-weight:900}
        .donut-meta{font-size:12px; color:#9ca3af}

        @media (max-width: 1024px){ .pill-row{grid-template-columns:repeat(3,minmax(0,1fr))} .tiles{grid-template-columns:repeat(2,minmax(0,1fr))} }
        @media (max-width: 640px){ .pill-row{grid-template-columns:repeat(2,minmax(0,1fr))} .tiles{grid-template-columns:repeat(1,minmax(0,1fr))} }

        /* Hydration popup */
        .hydration-popup{
            position:relative;
            padding:14px;
            border:2px solid #38bdf8;
            background:#0b1220;
            border-radius:14px;
            box-shadow:0 10px 30px rgba(0,0,0,.35);
            margin-top:12px;
        }
        .hydration-popup h4{margin:0 0 8px 0}
        .hydration-actions{display:flex; gap:8px; margin-top:8px}
        .btn{padding:8px 10px; border-radius:8px; border:1px solid #1e293b; background:#0f172a; color:#e5e7eb; cursor:pointer}
        .btn.primary{border-color:#1d4ed8; background:#1d4ed8;}

        /* Responsive utilities */
        @media (max-width: 1024px){
          .grid{grid-template-columns:repeat(6,minmax(0,1fr))}
          .col-6{grid-column:span 6 / span 6}
          .col-4{grid-column:span 6 / span 6}
          .col-3{grid-column:span 3 / span 3}
        }
        @media (max-width: 1200px){ .brand-title{font-size:42px} }
        @media (max-width: 900px){ .brand-title{font-size:38px} }
        @media (max-width: 640px){
          .grid{grid-template-columns:repeat(1,minmax(0,1fr))}
          .col-3,.col-4,.col-6{grid-column:span 1 / span 1}
          .brand-title{font-size:36px}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

# -------------------- DB / Auth --------------------
def sqlite3_connect():
    import sqlite3
    return sqlite3.connect(DB_PATH)

def init_db():
    with closing(sqlite3_connect()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE COLLATE NOCASE,
                email TEXT UNIQUE COLLATE NOCASE,
                salt TEXT NOT NULL,
                pw_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

def _hash_password(plain: str, salt_hex: str) -> str:
    return hashlib.sha256((salt_hex + plain).encode("utf-8")).hexdigest()

def create_user(username: str, email: str, password: str):
    username = username.strip()
    email = email.strip()
    if not re.match(r"^[A-Za-z0-9_.-]{3,20}$", username):
        return False, "Username must be 3–20 chars (letters, numbers, . _ -)."
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return False, "Enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    salt_hex = secrets.token_hex(16)
    pw_hash = _hash_password(password, salt_hex)
    try:
        with closing(sqlite3_connect()) as conn:
            conn.execute(
                "INSERT INTO users (username, email, salt, pw_hash) VALUES (?,?,?,?)",
                (username, email, salt_hex, pw_hash),
            )
            conn.commit()
        ensure_user_files(username)
        return True, "Account created! You can now log in."
    except Exception as e:
        msg = str(e).lower()
        if "username" in msg:
            return False, "That username is already taken."
        if "email" in msg:
            return False, "That email is already registered."
        return False, "Account could not be created."

def authenticate_user(username_or_email: str, password: str):
    q = username_or_email.strip()
    with closing(sqlite3_connect()) as conn:
        cur = conn.cursor()
        cur.execute("SELECT username, salt, pw_hash FROM users WHERE username = ? COLLATE NOCASE", (q,))
        row = cur.fetchone()
        if not row:
            cur.execute("SELECT username, salt, pw_hash FROM users WHERE email = ? COLLATE NOCASE", (q,))
            row = cur.fetchone()
        if not row:
            return False, None
        username, salt_hex, pw_hash_db = row
        ok = (_hash_password(password, salt_hex) == pw_hash_db)
        return ok, (username if ok else None)

init_db()

# -------------------- Files / Sessions / Badges --------------------
def get_user_log_path(username: str) -> str:
    safe_user = re.sub(r"[^A-Za-z0-9_.-]", "_", username)
    return os.path.join("user_logs", f"{safe_user}_exercise_log.csv")

def get_user_sessions_path(username: str) -> str:
    safe_user = re.sub(r"[^A-Za-z0-9_.-]", "_", username)
    return os.path.join("user_logs", f"{safe_user}_sessions.csv")

def get_user_badges_path(username: str) -> str:
    safe_user = re.sub(r"[^A-Za-z0-9_.-]", "_", username)
    return os.path.join("user_logs", f"{safe_user}_badges.json")

def ensure_user_files(username: str):
    rep_path = get_user_log_path(username)
    if not os.path.exists(rep_path):
        with open(rep_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Timestamp", "Exercise", "Reps"])
    sp = get_user_sessions_path(username)
    if not os.path.exists(sp):
        with open(sp, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["start_ts","end_ts","duration_s","total_reps","per_ex_json","calories"])
    bp = get_user_badges_path(username)
    if not os.path.exists(bp):
        with open(bp, "w", encoding="utf-8") as f:
            json.dump({"unlocked": [], "unlocked_at": {}}, f)

def live_status_path(username):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", username)
    return os.path.join("user_logs", f"{safe}_live.json")

def write_live_status(username, current_mode, counts):
    counts_simple = {k: int(v) for k,v in (counts or {}).items()}
    path = live_status_path(username)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json_obj = {"current": current_mode, "counts": counts_simple, "ts": time.time()}
            json.dump(json_obj, f)
        os.replace(tmp, path)
    except Exception:
        pass

def read_live_status(username):
    path = live_status_path(username)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None

def apply_dashboard_background_for_home():
    """Inject a subtle background image ONLY for the Home dashboard page.
    Uses images/dashboard_bg.jpg strictly.
    """
    uri = _image_to_data_uri("images/dashboard_bg.jpg")
    if not uri:
        return  # Do nothing if the specific file is not present
    st.markdown(
        f"""
        <style>
        /* Dashboard background image layer */
        body::before{{
          content:"";
          position:fixed; inset:0;
          background-image:url('{uri}');
          background-size:cover; background-position:center; background-repeat:no-repeat;
          opacity:.24; filter:saturate(1.05) contrast(1.05);
          pointer-events:none; z-index:-1;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def _video_to_data_uri(path: str):
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
                return f"data:video/mp4;base64,{b64}"
    except Exception:
        pass
    return None

def clear_live_status(username):
    try:
        os.remove(live_status_path(username))
    except Exception:
        pass

def _calories_for_counts(counts_dict):
    total = 0.0
    for k, v in counts_dict.items():
        total += float(v) * CALORIES_PER_REP.get(k, 0.0)
    return total

def append_session(username, start_ts, end_ts, counts_dict):
    ensure_user_files(username)
    sp = get_user_sessions_path(username)
    duration = max(0, int(end_ts - start_ts))
    total_reps = sum(int(v) for v in counts_dict.values())
    # ------------ UPDATED: calories from per-exercise multipliers ------------
    calories = int(round(_calories_for_counts(counts_dict)))
    per_ex_json = json.dumps(counts_dict)
    with open(sp, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts)),
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts)),
                                duration, total_reps, per_ex_json, calories])
    evaluate_and_unlock_badges(username)

# -------------------- Badges --------------------
BADGE_DEFS = {
    "rep_master_100": {
        "title": "Rep Master — 100 total reps",
        "type": "lifetime",
        "threshold": 100,
        "hint": "Accumulate 100 total reps across all sessions."
    },
    "rep_master_500": {
        "title": "Rep Master — 500 total reps",
        "type": "lifetime",
        "threshold": 500,
        "hint": "Accumulate 500 total reps across all sessions."
    },
    "session_50": {
        "title": "Session Beast — 50 reps in one session",
        "type": "session",
        "threshold": 50,
        "hint": "Complete 50 total reps during a single workout session."
    },
    "consistency_7": {
        "title": "7-Day Streak — 7 workouts in 7 days",
        "type": "consistency",
        "threshold": 7,
        "hint": "Work out at least once per day for 7 days."
    },
}

def load_badges(username):
    bp = get_user_badges_path(username)
    try:
        with open(bp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"unlocked": [], "unlocked_at": {}}

def save_badges(username, data):
    bp = get_user_badges_path(username)
    with open(bp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def sessions_list(username):
    sp = get_user_sessions_path(username)
    rows = []
    if os.path.exists(sp):
        with open(sp, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                rows.append(r)
    return rows

def total_reps_lifetime(username):
    rows = sessions_list(username)
    s = 0
    for r in rows:
        try:
            s += int(r.get("total_reps", 0))
        except Exception:
            pass
    return s

def total_calories_lifetime(username):
    rows = sessions_list(username)
    s = 0
    for r in rows:
        try:
            s += int(r.get("calories", 0))
        except Exception:
            pass
    return s

def best_session_reps(username):
    rows = sessions_list(username)
    best = 0
    for r in rows:
        try:
            best = max(best, int(r.get("total_reps", 0)))
        except Exception:
            pass
    return best

def workouts_in_last_days(username, days=7):
    rows = sessions_list(username)
    cutoff = datetime.now().date() - timedelta(days=days-1)
    count = 0
    for r in rows:
        try:
            d = datetime.strptime(r["start_ts"], "%Y-%m-%d %H:%M:%S").date()
            if d >= cutoff:
                count += 1
        except Exception:
            pass
    return count

def evaluate_and_unlock_badges(username):
    ensure_user_files(username)
    badges = load_badges(username)
    unlocked = set(badges.get("unlocked", []))

    total = total_reps_lifetime(username)
    for k,v in BADGE_DEFS.items():
        if v["type"] == "lifetime" and total >= v["threshold"]:
            if k not in unlocked:
                unlocked.add(k)
                badges["unlocked_at"][k] = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = sessions_list(username)
    if rows:
        last = rows[-1]
        try:
            last_reps = int(last.get("total_reps", 0))
            for k,v in BADGE_DEFS.items():
                if v["type"] == "session" and last_reps >= v["threshold"]:
                    if k not in unlocked:
                        unlocked.add(k)
                        badges["unlocked_at"][k] = time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    cnt7 = workouts_in_last_days(username, days=7)
    if cnt7 >= BADGE_DEFS["consistency_7"]["threshold"]:
        k="consistency_7"
        if k not in unlocked:
            unlocked.add(k)
            badges["unlocked_at"][k] = time.strftime("%Y-%m-%d %H:%M:%S")

    badges["unlocked"] = sorted(list(unlocked))
    save_badges(username, badges)
    return badges

# -------------------- Audio --------------------
sound_lock = threading.Lock()
def speak_file(path):
    if not audio_enabled:
        return
    with sound_lock:
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
        except Exception:
            pass

def speak_text(name, text):
    if not audio_enabled:
        return
    path = f"sounds/{name}.mp3"
    if not os.path.exists(path):
        try:
            gTTS(text=text, lang="en").save(path)
        except Exception:
            return
    speak_file(path)

if audio_enabled and not os.path.exists("sounds/motivation.mp3"):
    try:
        gTTS("Great job! Keep going!").save("sounds/motivation.mp3")
    except Exception:
        pass

# -------------------- Exercise detection (kept) --------------------
exercise_modes = ["squat", "pushup", "jumping_jack", "high_knees", "arm_raise"]

def calc_angle(a, b, c):
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba, bc = a - b, c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return 180.0
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

# -------------------- Video transformer (kept) --------------------
if VideoTransformerBase is object:
    class ExerciseTransformer(object):
        def __init__(self, username=None, perf_mode: bool=False):
            self.username = username
            # Add frame buffer for smoother video
            self.frame_buffer = deque(maxlen=1 if perf_mode else 2)
            self.last_frame_time = time.time()
            self.target_fps = 24 if perf_mode else 30
            self.frame_interval = 1.0 / self.target_fps
else:
    class ExerciseTransformer(VideoTransformerBase):
        def __init__(self, username=None, perf_mode: bool=False):
            self.username = username
            self.mp_pose = mp.solutions.pose
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_hands = mp.solutions.hands

            self.mode_index = 0
            self.current_mode = exercise_modes[self.mode_index]
            self.rep_counts = {m: 0 for m in exercise_modes}
            self.stage = None
            self.last_rep_time = 0
            self.min_rep_interval = 0.6

            self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
            self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

            self.peace_buffer = 0
            self.last_switch_time = 0

            self.buf_angles = {"knee": deque(maxlen=5), "elbow": deque(maxlen=5)}
            self.buf_y = {k: deque(maxlen=5) for k in ["lw","rw","ls","rs","lk","lh","rk","rh","la","ra"]}
            self.last_knee_up = None
            
            # Add frame buffer for smoother video
            self.frame_buffer = deque(maxlen=1 if perf_mode else 2)
            self.last_frame_time = time.time()
            self.target_fps = 24 if perf_mode else 30
            self.frame_interval = 1.0 / self.target_fps

        # ... rest of your ExerciseTransformer methods unchanged ...

        def _pt(self, lm, name, h, w):
            try:
                idx = self.mp_pose.PoseLandmark[name].value
            except Exception:
                return None
            L = lm[idx]
            if getattr(L, "visibility", 1.0) < 0.5:
                return None
            return [L.x * w, L.y * h]

        def _safe_push(self, key, value):
            if value is not None:
                if isinstance(value, (list, tuple)):
                    self.buf_y[key].append(value[1])
                else:
                    self.buf_angles[key].append(value)

        @staticmethod
        def _median(arr):
            if not arr:
                return None
            return float(np.median(np.array(arr, dtype=float)))

        def detect_peace(self, hl):
            coords = hl.landmark
            H = self.mp_hands.HandLandmark
            idx_up = coords[H.INDEX_FINGER_TIP].y < coords[H.INDEX_FINGER_PIP].y
            mid_up = coords[H.MIDDLE_FINGER_TIP].y < coords[H.MIDDLE_FINGER_PIP].y
            ring_down = coords[H.RING_FINGER_TIP].y > coords[H.RING_FINGER_PIP].y
            pinky_down = coords[H.PINKY_TIP].y > coords[H.PINKY_PIP].y
            return idx_up and mid_up and ring_down and pinky_down

        def check_exercise(self, landmarks, h, w):
            m = self.current_mode
            down = False
            def knee_angle():
                hipL = self._pt(landmarks, "LEFT_HIP", h, w)
                kneeL = self._pt(landmarks, "LEFT_KNEE", h, w)
                ankleL = self._pt(landmarks, "LEFT_ANKLE", h, w)
                hipR = self._pt(landmarks, "RIGHT_HIP", h, w)
                kneeR = self._pt(landmarks, "RIGHT_KNEE", h, w)
                ankleR = self._pt(landmarks, "RIGHT_ANKLE", h, w)
                angs = []
                if hipL and kneeL and ankleL: angs.append(calc_angle(hipL, kneeL, ankleL))
                if hipR and kneeR and ankleR: angs.append(calc_angle(hipR, kneeR, ankleR))
                if not angs: return None
                return min(angs)

            def elbow_angle():
                shL = self._pt(landmarks, "LEFT_SHOULDER", h, w)
                elL = self._pt(landmarks, "LEFT_ELBOW", h, w)
                wrL = self._pt(landmarks, "LEFT_WRIST", h, w)
                shR = self._pt(landmarks, "RIGHT_SHOULDER", h, w)
                elR = self._pt(landmarks, "RIGHT_ELBOW", h, w)
                wrR = self._pt(landmarks, "RIGHT_WRIST", h, w)
                angs = []
                if shL and elL and wrL: angs.append(calc_angle(shL, elL, wrL))
                if shR and elR and wrR: angs.append(calc_angle(shR, elR, wrR))
                if not angs: return None
                return min(angs)

            if m == "squat":
                ang = knee_angle()
                self._safe_push("knee", ang)
                ang_med = self._median(self.buf_angles["knee"]) if self.buf_angles["knee"] else ang
                if ang_med is not None:
                    down = ang_med < (95 if self.stage != "down" else 160)
            elif m == "pushup":
                ang = elbow_angle()
                self._safe_push("elbow", ang)
                ang_med = self._median(self.buf_angles["elbow"]) if self.buf_angles["elbow"] else ang
                if ang_med is not None:
                    down = ang_med < (95 if self.stage != "down" else 155)
            elif m == "jumping_jack":
                lw = self._pt(landmarks, "LEFT_WRIST", h, w)
                rw = self._pt(landmarks, "RIGHT_WRIST", h, w)
                ls = self._pt(landmarks, "LEFT_SHOULDER", h, w)
                rs = self._pt(landmarks, "RIGHT_SHOULDER", h, w)
                for k, v in zip(["lw","rw","ls","rs"], [lw,rw,ls,rs]):
                    if v is not None: self._safe_push(k, v)
                def med(k, raw): return self._median(self.buf_y[k]) if self.buf_y[k] else (raw[1] if raw else None)
                lw_y, rw_y, ls_y, rs_y = med("lw", lw), med("rw", rw), med("ls", ls), med("rs", rs)
                if None not in (lw_y, rw_y, ls_y, rs_y):
                    arms_up = lw_y < ls_y and rw_y < rs_y
                    down = arms_up if self.stage != "down" else not (lw_y > ls_y and rw_y > rs_y)
            elif m == "high_knees":
                lk = self._pt(landmarks, "LEFT_KNEE", h, w)
                lh = self._pt(landmarks, "LEFT_HIP", h, w)
                rk = self._pt(landmarks, "RIGHT_KNEE", h, w)
                rh = self._pt(landmarks, "RIGHT_HIP", h, w)
                for k, v in zip(["lk","lh","rk","rh"], [lk,lh,rk,rh]):
                    if v is not None: self._safe_push(k, v)
                def med(k, raw): return self._median(self.buf_y[k]) if self.buf_y[k] else (raw[1] if raw else None)
                lk_y, lh_y, rk_y, rh_y = med("lk", lk), med("lh", lh), med("rk", rk), med("rh", rh)
                if None not in (lk_y, lh_y, rk_y, rh_y):
                    left_up = lk_y < lh_y - 10
                    right_up = rk_y < rh_y - 10
                    if left_up and self.last_knee_up != "L":
                        down = True; self.last_knee_up = "L"
                    elif right_up and self.last_knee_up != "R":
                        down = True; self.last_knee_up = "R"
                    else:
                        down = left_up or right_up
            elif m == "arm_raise":
                lw = self._pt(landmarks, "LEFT_WRIST", h, w)
                rw = self._pt(landmarks, "RIGHT_WRIST", h, w)
                ls = self._pt(landmarks, "LEFT_SHOULDER", h, w)
                rs = self._pt(landmarks, "RIGHT_SHOULDER", h, w)
                for k, v in zip(["lw","rw","ls","rs"], [lw,rw,ls,rs]):
                    if v is not None: self._safe_push(k, v)
                def med(k, raw): return self._median(self.buf_y[k]) if self.buf_y[k] else (raw[1] if raw else None)
                lw_y, rw_y, ls_y, rs_y = med("lw", lw), med("rw", rw), med("ls", ls), med("rs", rs)
                if None not in (lw_y, rw_y, ls_y, rs_y):
                    arms_up = lw_y < ls_y and rw_y < rs_y
                    down = arms_up if self.stage != "down" else not (lw_y > ls_y and rw_y > rs_y)
            return down

        def __init__(self, username=None, perf_mode: bool=False):
            super().__init__()
            self.username = username
            self.mp_pose = mp.solutions.pose
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_hands = mp.solutions.hands

            self.mode_index = 0
            self.current_mode = exercise_modes[self.mode_index]
            self.rep_counts = {m: 0 for m in exercise_modes}
            self.stage = None
            self.last_rep_time = 0
            self.min_rep_interval = 0.6

            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
                model_complexity=(0 if perf_mode else 1)  # Toggle for performance mode
            )
            self.hands = self.mp_hands.Hands(
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8,
                max_num_hands=1  # Limit to one hand for better performance
            )

            self.peace_buffer = 0
            self.last_switch_time = 0

            self.buf_angles = {"knee": deque(maxlen=5), "elbow": deque(maxlen=5)}
            self.buf_y = {k: deque(maxlen=5) for k in ["lw","rw","ls","rs","lk","lh","rk","rh","la","ra"]}
            self.last_knee_up = None
            
            # Add frame buffer for smoother video
            self.frame_buffer = deque(maxlen=2)
            self.last_frame_time = time.time()
            self.target_fps = 30
            self.frame_interval = 1.0 / self.target_fps

        def recv(self, frame: "VideoFrame") -> "VideoFrame":
            current_time = time.time()
            
            # Skip frames if we're processing too fast
            if current_time - self.last_frame_time < self.frame_interval:
                return frame
                
            self.last_frame_time = current_time
            
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            
            # Add frame to buffer
            self.frame_buffer.append(img.copy())
            if len(self.frame_buffer) < 2:
                return frame
                
            # Get the most recent frame from buffer
            img = self.frame_buffer[-1]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            hand_res = self.hands.process(rgb)
            if hand_res.multi_hand_landmarks:
                for hl in hand_res.multi_hand_landmarks:
                    if self.detect_peace(hl):
                        self.peace_buffer += 1
                    else:
                        self.peace_buffer = 0
                    self.mp_draw.draw_landmarks(img, hl, self.mp_hands.HAND_CONNECTIONS)

            if self.peace_buffer > 5 and (time.time() - self.last_switch_time > 2):
                self.mode_index = (self.mode_index + 1) % len(exercise_modes)
                self.current_mode = exercise_modes[self.mode_index]
                if self.username:
                    write_live_status(self.username, self.current_mode, self.rep_counts)
                speak_text(self.current_mode, f"Starting {self.current_mode}")
                self.stage = None
                self.last_switch_time = time.time()
                self.peace_buffer = 0
                for k in self.buf_angles: self.buf_angles[k].clear()
                for k in self.buf_y: self.buf_y[k].clear()
                self.last_knee_up = None

            pose_res = self.pose.process(rgb)
            if pose_res.pose_landmarks:
                down = self.check_exercise(pose_res.pose_landmarks.landmark, h, w)
                now = time.time()
                if down and self.stage != "down":
                    self.stage = "down"
                elif not down and self.stage == "down":
                    if (now - self.last_rep_time) > self.min_rep_interval:
                        self.last_rep_time = now
                        self.rep_counts[self.current_mode] += 1
                        if self.username:
                            write_live_status(self.username, self.current_mode, self.rep_counts)
                        speak_text(str(self.rep_counts[self.current_mode]), str(self.rep_counts[self.current_mode]))
                        if self.rep_counts[self.current_mode] % 10 == 0:
                            speak_file("sounds/motivation.mp3")
                    self.stage = "up"

                self.mp_draw.draw_landmarks(img, pose_res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            cv2.putText(img, f"Current: {self.current_mode}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            return VideoFrame.from_ndarray(img, format="bgr24")

# -------------------- UI Helpers & Pages --------------------
def show_topbar():
    user = st.session_state.get("auth_user")
    user_html = (
        f"<span class='pill user-pill' style='font-size:18px; font-weight:600;'>Logged in: {user}</span>"
        if user else ""
    )
    st.markdown(
        f"""
        <div class='topbar'>
          <div class='brand-title' style="text-transform:uppercase; margin-bottom:0;">
            <span class='next'>Next</span><span class='gen'>Gen</span> Fitness Tracker 🏋
          </div>
          {user_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def show_navbar_sidebar():
    st.sidebar.title("🏋 Gym Tracker Navigation")

    # Profile snapshot
    user = st.session_state.get("auth_user")
    if user:
        st.sidebar.markdown(
            f"""
            <div class='card soft' style='margin-bottom:10px;'>
                <div style='display:flex; align-items:center; gap:12px;'>
                    <div style='width:44px; height:44px; border-radius:9999px; background:#1f2937; display:flex; align-items:center; justify-content:center; font-weight:800;'>
                        {user[:1].upper()}
                    </div>
                    <div>
                        <div style='font-weight:800'>{user}</div>
                        <div class='subtext'>Ready to train</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Default sidebar navigation
    if st.session_state.get("auth_user"):
        options = ["Home", "Workout", "Achievements", "Reports", "Logout"]
    else:
        options = ["Login / Sign up"]

    # preserve last selected page (default Home)
    cur_page = st.session_state.get("current_page", "Home")
    if cur_page not in options:
        cur_page = options[0]

    # compute index so radio highlights current_page
    try:
        default_index = options.index(cur_page)
    except Exception:
        default_index = 0

    selected = st.sidebar.radio("Go to", options, index=default_index)

    # If Start Workout button forced navigation
    if st.session_state.get("force_page"):
        selected = st.session_state["force_page"]
        st.session_state["force_page"] = None  # reset after use

    # save current page to session so next render highlights properly
    st.session_state["current_page"] = selected
    return selected


def daily_quote():
    quotes = [
        "Do something today that your future self will thank you for.",
        "Small progress is still progress.",
        "Consistency beats intensity. Keep going.",
        "One more rep counts.",
        "Focus on the process, not perfection."
    ]
    idx = int(time.time() // 5) % len(quotes)
    return quotes[idx]

def compute_consecutive_streak(username):
    rows = sessions_list(username)
    if not rows:
        return 0
    days = sorted({ datetime.strptime(r["start_ts"], "%Y-%m-%d %H:%M:%S").date() for r in rows }, reverse=True)
    streak = 0
    today = datetime.now().date()
    day = today
    for d in days:
        if d == day:
            streak += 1
            day = day - timedelta(days=1)
        elif d < day:
            if streak==0 and d==today - timedelta(days=1):
                streak += 1
                day = day - timedelta(days=1)
            else:
                break
    return streak

def _stat_card(icon, label, value, subtext=""):
    return f"""
    <div class='card soft'>
      <div class='title-xs'>{icon} {label}</div>
      <div class='value-xl'>{value}</div>
      <div class='subtext'>{subtext}</div>
    </div>
    """

def _achv_card(emoji, name, done, need):
    pct = 0 if need <= 0 else min(100, int(done * 100 / need))
    return f"""
    <div class='achv'>
      <div class='left'>{emoji}</div>
      <div class='meta'>
        <div class='name'>{name}</div>
        <div class='subtext'>Progress: {done} / {need}</div>
        <div class='bar red'><span style='width:{pct}%;'></span></div>
      </div>
    </div>
    """

def page_home():
    user = st.session_state.get("auth_user")
    ensure_user_files(user)
    rows = sessions_list(user)
    total_sessions = len(rows)
    total_reps = total_reps_lifetime(user)
    total_cal = total_calories_lifetime(user)
    best = best_session_reps(user)

    # Compute today's stats and 7-day series
    today = datetime.now().date()
    per_day = {}
    today_reps = 0
    for r in rows:
        try:
            d = datetime.strptime(r["start_ts"], "%Y-%m-%d %H:%M:%S").date()
            total = int(r.get("total_reps", 0))
            per_day[d] = per_day.get(d, 0) + total
            if d == today:
                today_reps += total
        except Exception:
            pass

    # Header KPIs (summary cards)
    st.markdown("<div class='grid'>", unsafe_allow_html=True)
    _kpi_today = _stat_card('🎯', "Today's reps", today_reps, 'Daily progress')
    st.markdown(f"<div class='col-3'>{_kpi_today}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='col-3'>{_stat_card('⏱','Sessions', total_sessions, 'All-time')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='col-3'>{_stat_card('🔥','Calories', total_cal, 'Estimated')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='col-3'>{_stat_card('🏆','Best Reps', best, 'Single session')}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Pill row (Today plan, Sleep, Steps, Food, Heart)
    st.markdown("<div class='panel' style='margin-top:10px'>", unsafe_allow_html=True)
    st.markdown("<div class='pill-row'>", unsafe_allow_html=True)
    pills = [
        {"ic":"🟢","ttl":"Today's plan","sub":"4 of 5 goals","val":"80%"},
        {"ic":"😴","ttl":"Sleep","sub":"6.5 hrs","val":""},
        {"ic":"👣","ttl":"Steps","sub":"8200","val":""},
        {"ic":"🍎","ttl":"Food","sub":"1250 kcal","val":""},
        {"ic":"❤","ttl":"Heart","sub":"63 bpm","val":""},
    ]
    for p in pills:
        st.markdown(
            f"""
            <div class='pill-card'>
              <div class='pill-ic'>{p['ic']}</div>
              <div class='pill-meta'><div class='ttl'>{p['ttl']}</div><div class='sub'>{p['sub']}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Main grid: Activity line + Goals + Start CTA
    st.markdown("<div class='grid' style='margin-top:10px;'>", unsafe_allow_html=True)
    # Activity line (col-6)
    with st.container():
        st.markdown("<div class='col-6'>", unsafe_allow_html=True)
        # build dataframe for last 7 days
        dates = [today - timedelta(days=i) for i in range(6,-1,-1)]
        df = pd.DataFrame({
            "date": dates,
            "reps": [per_day.get(d, 0) for d in dates]
        })
        line = (
            alt.Chart(df)
            .mark_line(color="#22d3ee", point=True)
            .encode(x=alt.X("date:T", title="Date"), y=alt.Y("reps:Q", title="Reps"))
            .properties(height=220)
        )
        st.altair_chart(line, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Goals (col-3)
    with st.container():
        st.markdown("<div class='col-3'>", unsafe_allow_html=True)
        st.markdown("<div class='title-xs'>Your Goals</div>", unsafe_allow_html=True)
        def bar(label, done, total):
            pct = 0 if total<=0 else min(100, int(done*100/total))
            return f"<div style='margin:10px 0'><div style='font-size:12px;color:#94a3b8'>{label}</div><div class='bar red'><span style='width:{pct}%;'></span></div></div>"
        # Sample goals using today's/weekly reps
        weekly_target = 300
        st.markdown(bar("Weekly Reps", int(df["reps"].sum()), weekly_target), unsafe_allow_html=True)
        st.markdown(bar("Today Reps", today_reps, 100), unsafe_allow_html=True)
        st.markdown(bar("Calories", total_cal % 500, 500), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # CTA / tiles (col-3)
    with st.container():
        st.markdown("<div class='col-3'>", unsafe_allow_html=True)
        st.markdown("<div class='title-xs'>Workout</div>", unsafe_allow_html=True)
        # Tiles
        def _tile(title, img_name=None):
            uri = _image_to_data_uri(f"images/{img_name}") if img_name else None
            if uri:
                return f"<div class='w-tile'><img class='w-img' src='{uri}'/><div class='w-cap'>{title}</div></div>"
            else:
                return f"<div class='w-tile'><div class='w-img' style='background:linear-gradient(135deg,#0f172a,#1e293b)'></div><div class='w-cap'>{title}</div></div>"
        tiles_html = "<div class='tiles'>" + \
            _tile("Squats","squats.jpg") + \
            _tile("Push-ups","pushups.jpg") + \
            _tile("Jumping Jacks","jumping_jacks.jpg") + \
            _tile("High Knees","high_knees.jpg") + \
            _tile("Arm Raise","arm_raise.jpg") + \
            "</div>"
        st.markdown(tiles_html, unsafe_allow_html=True)
        # Start button
        if st.button("🟢 Start Workout", use_container_width=True):
            st.session_state["force_page"] = "Workout"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Donut stats row
    st.markdown("<div class='grid' style='margin-top:10px;'>", unsafe_allow_html=True)
    st.markdown("<div class='col-6'><div class='panel'><div class='title-xs'>Workout Statistics</div>", unsafe_allow_html=True)
    # Simple donut using CSS var
    weekly_minutes = int(sum([per_day.get(today - timedelta(days=i),0) for i in range(7)]) // 10)
    pct = min(100, int((weekly_minutes/60)*100))
    st.markdown(
        f"""
        <div class='donut-wrap'>
            <div class='donut-box'>
                <div class='donut' style='--p:{pct};'></div>
                <div class='donut-val'>{weekly_minutes}m</div>
            </div>
            <div>
                <div class='title-xs'>This week</div>
                <div class='donut-meta'>Target: 60 min</div>
                <div class='donut-meta'>Calories est: {total_cal}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"*Motivation:* {daily_quote()}")

# -------------------- WORKOUT (fixed) --------------------
def page_workout():
    st.markdown("## 💪 Workout")
    st.caption("Show ✌ (peace) gesture to switch exercises. Use the Start/Stop button to control your session and camera.")

    if webrtc_streamer is None or VideoFrame is None:
        st.error("Install streamlit-webrtc and av to use camera: pip install streamlit-webrtc av")
        return
    if cv2 is None:
        st.error("Install OpenCV: pip install opencv-python")
        return
    if mp is None:
        st.warning("MediaPipe not installed; camera tracking requires mediapipe.")
        return

    user = st.session_state.get("auth_user")
    if not user:
        st.warning("Please login to use the tracker.")
        return

    # --- session-state defaults ---
    ss = st.session_state
    ss.setdefault("workout_active", False)
    ss.setdefault("workout_start_time", None)
    ss.setdefault("workout_initial_counts", {m:0 for m in exercise_modes})

    # Hydration controls/state
    ss.setdefault("last_hydration", time.time())
    ss.setdefault("hydration_interval_s", 20*60)  # 20 minutes
    ss.setdefault("last_hydration_toast", 0.0)
    # NEW: popup control
    ss.setdefault("hydration_popup_until", 0.0)  # timestamp when popup should auto-hide
    ss.setdefault("hydration_voice_fired", False)

    col1, col2 = st.columns([2, 1])

    # ================= LEFT SIDE =================
    with col1:
        st.markdown("""
        <div class="camera-shell">
          <div class="camera-title">
            <span>📷 Live Feed — Pose Tracking</span>
            <span class="pill-rec">● REC</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Preflight readiness
        ready_msgs = []
        ready = True
        if webrtc_streamer is None or VideoFrame is None:
            ready = False; ready_msgs.append("❌ Camera module not installed")
        else:
            ready_msgs.append("✅ Camera module ready")
        if cv2 is None:
            ready = False; ready_msgs.append("❌ OpenCV missing")
        else:
            ready_msgs.append("✅ OpenCV ready")
        if mp is None:
            ready = False; ready_msgs.append("❌ MediaPipe missing")
        else:
            ready_msgs.append("✅ MediaPipe ready")
        st.markdown("<br>" + "<br>".join(ready_msgs), unsafe_allow_html=True)

        # ONE unified button that starts/stops everything
        start_stop = st.button(("🟢 Start Workout" if not ss["workout_active"] else "⛔ Stop Workout"), use_container_width=True, disabled=(not ready and not ss["workout_active"]))

        if start_stop and not ss["workout_active"]:
            # start
            ss["workout_active"] = True
            ss["workout_start_time"] = time.time()
            ss["workout_initial_counts"] = {m:0 for m in exercise_modes}
            ensure_user_files(user)
            write_live_status(user, exercise_modes[0], ss["workout_initial_counts"])
            # reset hydration on workout start
            ss["last_hydration"] = time.time()
            ss["last_hydration_toast"] = 0.0
            ss["hydration_popup_until"] = 0.0
            ss["hydration_voice_fired"] = False

        elif start_stop and ss["workout_active"]:
            # stop & save
            final = read_live_status(user)
            final_counts = {m:int(final["counts"].get(m,0)) for m in exercise_modes} if final else {m:0 for m in exercise_modes}
            start_ts = ss["workout_start_time"] or time.time()
            initial = ss.get("workout_initial_counts", {m:0 for m in exercise_modes})
            delta = {m:max(0, final_counts.get(m,0)-initial.get(m,0)) for m in exercise_modes}
            append_session(user, start_ts, time.time(), delta)
            clear_live_status(user)
            ss["workout_active"] = False
            ss["workout_start_time"] = None
            st.success("Workout session saved ✅")

        # Camera feed (only when active)
        if ss["workout_active"]:
            # Add some CSS to prevent layout shifts
            st.markdown("""
            <style>
                .stVideo {
                    border-radius: 8px;
                    overflow: hidden;
                }
                .stVideo > video {
                    width: 100% !important;
                    height: auto !important;
                }
            </style>
            """, unsafe_allow_html=True)
            # Build RTC configuration with optional TURN
            ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
            if ss.get("turn_url"):
                turn_entry = {"urls": [ss.get("turn_url").strip()]}
                if ss.get("turn_username"):
                    turn_entry["username"] = ss.get("turn_username").strip()
                if ss.get("turn_credential"):
                    turn_entry["credential"] = ss.get("turn_credential").strip()
                ice_servers.append(turn_entry)
            rtc_configuration = {"iceServers": ice_servers}
            
            # Add a container with fixed dimensions to prevent layout shifts
            video_container = st.container()
            with video_container:
                ctx = webrtc_streamer(
                    key=f"exercise_{user}",
                    video_processor_factory=lambda: ExerciseTransformer(username=user, perf_mode=ss.get("perf_mode", False)),
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": 640},
                            "height": {"ideal": 480},
                            "frameRate": {"ideal": (24 if ss.get("perf_mode", False) else 30), "min": 20}
                        },
                        "audio": False
                    },
                    rtc_configuration=rtc_configuration,
                    async_processing=True,
                    video_html_attrs={
                        "style": {
                            "width": "100%",
                            "margin": "0 auto",
                            "border": "2px solid #3b82f6",
                            "borderRadius": "8px"
                        },
                        "controls": False,
                        "autoPlay": True,
                        "muted": True
                    }
                )
                # Basic connection status to aid troubleshooting
                try:
                    st.caption(f"RTC Status: {'playing' if getattr(ctx.state, 'playing', False) else 'not playing'}")
                except Exception:
                    pass

    # ================= RIGHT SIDE =================
    with col2:
        st.markdown("### 📊 Current Exercise")

        # Performance mode toggle
        ss.setdefault("perf_mode", False)
        ss["perf_mode"] = st.toggle("Performance mode (lower quality, faster)", value=ss["perf_mode"])

        # Advanced WebRTC options (for networks with strict policies)
        with st.expander("Advanced WebRTC (TURN)"):
            ss.setdefault("turn_url", "")
            ss.setdefault("turn_username", "")
            ss.setdefault("turn_credential", "")
            ss["turn_url"] = st.text_input("TURN URL (e.g., turn:your.turn.host:3478)", value=ss["turn_url"])
            c1, c2 = st.columns(2)
            with c1:
                ss["turn_username"] = st.text_input("TURN Username", value=ss["turn_username"])
            with c2:
                ss["turn_credential"] = st.text_input("TURN Credential/Password", value=ss["turn_credential"], type="password")

        data = read_live_status(user)
        if data:
            current = data.get("current", exercise_modes[0])
            counts = data.get("counts", {m:0 for m in exercise_modes})
        else:
            current, counts = exercise_modes[0], {m:0 for m in exercise_modes}

        # Use the SAME ids as detector so counts match
        exercises = [
            {"id":"squat",        "name":"Squats",         "target":50, "icon":"🏋"},
            {"id":"pushup",       "name":"Push-ups",       "target":30, "icon":"💪"},
            {"id":"jumping_jack", "name":"Jumping Jacks", "target":40, "icon":"🤸‍♂️"},
            {"id":"high_knees",  "name":"High Knees",     "target":60, "icon":"🤾‍♂️"},
            {"id":"arm_raise",   "name":"Arm Raise",      "target":40, "icon":"🙆‍♂️"}

        ]
        for ex in exercises:
            active = "border:2px solid #3b82f6; background:#1e3a8a10;" if current==ex["id"] else "border:1px solid #334155;"
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:center; 
                            padding:8px 12px; margin:6px 0; border-radius:8px; {active}">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span style="font-size:20px;">{ex["icon"]}</span>
                        <div>
                            <div style="font-weight:600;">{ex["name"]}</div>
                            <div style="font-size:12px; color:#64748b;">Target: {ex["target"]}</div>
                        </div>
                    </div>
                    <div style="font-weight:bold; color:#3b82f6;">{int(counts.get(ex["id"],0))}/{ex["target"]}</div>
                </div>
                """, unsafe_allow_html=True)

        # Progress
        target = next((e["target"] for e in exercises if e["id"]==current), 30)
        reps_done = int(counts.get(current,0))
        pct = min(100, int(reps_done*100/target)) if target else 0

        st.markdown(f"""
        <h4 style="margin-top:20px;">Progress</h4>
        <div style="font-size:13px; color:#94a3b8;">Reps Completed</div>
        <div style="margin-bottom:6px; font-weight:bold; color:#3b82f6;">{reps_done}/{target}</div>
        <div style="height:8px; border-radius:6px; background:#1f2937;">
            <div style="width:{pct}%; height:8px; background:linear-gradient(to right,#3b82f6,#2563eb); border-radius:6px;"></div>
        </div>
        """, unsafe_allow_html=True)

        # Duration + Calories (live)
        duration = int(time.time() - ss["workout_start_time"]) if ss["workout_active"] and ss["workout_start_time"] else 0
        mins, secs = divmod(duration, 60)

        # Live calories = (current counts - initial) * per-ex calories
        initial = ss.get("workout_initial_counts", {m:0 for m in exercise_modes})
        deltas = {m: max(0, int(counts.get(m,0)) - int(initial.get(m,0))) for m in exercise_modes}
        live_cals = int(round(_calories_for_counts(deltas)))

        st.markdown(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:12px;">
            <div class="tile-dark">
                <div class="big">{mins:02}:{secs:02}</div>
                <div class="label">Duration</div>
            </div>
            <div class="tile-dark">
                <div class="big">🔥 {live_cals}</div>
                <div class="label">Calories Burned</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Hydration reminder logic
                
        ss.setdefault("last_hydration", time.time())
        ss.setdefault("hydration_interval_s", 15*60)
        ss.setdefault("last_hydration_toast", 0.0)
        ss.setdefault("hydration_popup_until", 0.0)
        ss.setdefault("hydration_voice_fired", False)

        # compute elapsed since last drink
        elapsed = time.time() - ss["last_hydration"]

        # If due, popup + voice + toast
        if elapsed >= ss["hydration_interval_s"]:
            if time.time() > ss["hydration_popup_until"]:
                ss["hydration_popup_until"] = time.time() + 5
                ss["hydration_voice_fired"] = False

            if not ss["hydration_voice_fired"]:
                speak_text("drink_water", "Drink Water!")
                ss["hydration_voice_fired"] = True

            if time.time() - ss["last_hydration_toast"] > 60:
                st.toast("💧 Time to hydrate! Take a sip of water.", icon="💧")
                ss["last_hydration_toast"] = time.time()

            if time.time() <= ss["hydration_popup_until"]:
                st.markdown(
                    """
                    <div class="hydration-popup">
                        <h4>💧 Hydration Reminder</h4>
                        <div>It's time to drink water!</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("I drank ✅", key="hydr_ok"):
                        ss["last_hydration"] = time.time()
                        ss["last_hydration_toast"] = 0.0
                        ss["hydration_popup_until"] = 0.0
                        ss["hydration_voice_fired"] = False
                        st.toast("Hydration timer reset ✅", icon="✅")
                        st.rerun()
                with c2:
                    if st.button("Snooze 5 min ⏰", key="hydr_snooze"):
                        ss["last_hydration"] = time.time() - (ss["hydration_interval_s"] - 5*60)
                        ss["hydration_popup_until"] = 0.0
                        ss["hydration_voice_fired"] = False
                        st.toast("Snoozed for 5 minutes.", icon="⏰")
                        st.rerun()

        # Inline banner -> show elapsed time in MM:SS
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        hydra_msg = f"⏱ Elapsed since last drink: {mins:02d}:{secs:02d}"

        cA, cB = st.columns([1, 1])
        with cA:
            if st.button("Reset Hydration Timer", key="hydr_reset"):
                ss["last_hydration"] = time.time()
                ss["last_hydration_toast"] = 0.0
                ss["hydration_popup_until"] = 0.0
                ss["hydration_voice_fired"] = False
                elapsed = 0  # reset elapsed immediately
                st.toast("Hydration timer reset ✅", icon="✅")
                st.rerun()


        with cB:
            new_interval = st.selectbox(
                "Reminder every", [10, 15, 20, 30],
                index=[10, 15, 20, 30].index(int(ss["hydration_interval_s"]/60))
            )
            ss["hydration_interval_s"] = new_interval * 60

        st.markdown(
            f"<div style='margin-top:8px; background:#e0f2fe20; border:1px solid #1e40af55; padding:10px; border-radius:8px; font-size:13px; color:#93c5fd;'>{hydra_msg}</div>",
            unsafe_allow_html=True
        )

        if ss.get("workout_active", False):
            time.sleep(1)
            st.rerun()


# Achievements page 

def try_load_lottie_url(url: str):
    if requests is None:
        return None
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

LOTTIES = {
    "rep_master_100": try_load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_jbrw3hcz.json"),
    "rep_master_500": try_load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_touohxv0.json"),
    "session_50": try_load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_mjlh3hcy.json"),
    "consistency_7": try_load_lottie_url("https://assets8.lottiefiles.com/packages/lf20_x62chJ.json"),
}

def page_achievements():
    st.title("🏅 Achievements")
    user = st.session_state.get("auth_user")
    ensure_user_files(user)
    badges = load_badges(user)
    focus = st.session_state.get("focus_badge")
    if focus:
        st.markdown(f"### 🔎 Details — {BADGE_DEFS.get(focus,{}).get('title', focus)}")
        st.write(BADGE_DEFS.get(focus,{}).get("hint", "Requirement not set."))
        if BADGE_DEFS.get(focus,{}).get("type") == "lifetime":
            total = total_reps_lifetime(user)
            th = BADGE_DEFS[focus]["threshold"]
            st.info(f"Progress: {total} / {th} total reps")
        elif BADGE_DEFS.get(focus,{}).get("type") == "session":
            best = best_session_reps(user)
            th = BADGE_DEFS[focus]["threshold"]
            st.info(f"Best session: {best} / {th} reps")
        elif BADGE_DEFS.get(focus,{}).get("type") == "consistency":
            cnt = workouts_in_last_days(user, days=7)
            th = BADGE_DEFS[focus]["threshold"]
            st.info(f"Workouts in last 7 days: {cnt} / {th}")
        st.markdown("---")
        if "focus_badge" in st.session_state:
            del st.session_state["focus_badge"]

    st.subheader("All Achievements")
    unlocked = set(badges.get("unlocked", []))
    for k,info in BADGE_DEFS.items():
        title = info["title"]
        hint = info.get("hint", "")
        is_unlocked = k in unlocked
        col1, col2 = st.columns([1,1])
        with col1:
            if is_unlocked:
                st.markdown(f"<div class='card soft'><strong>{title}</strong><br><small>✅ Unlocked: {badges['unlocked_at'].get(k,'')}</small></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='card'><strong>{title}</strong><br><small>🔒 Locked — {hint}</small></div>", unsafe_allow_html=True)
        with col2:
            if st_lottie and LOTTIES.get(k):
                st_lottie(LOTTIES[k], height=90, key=f"lottie_{k}")
            else:
                emoji = "🏅" if is_unlocked else "🔒"
                st.markdown(f"{emoji} <span class='hint'>{hint}</span>", unsafe_allow_html=True)
            if info["type"] == "lifetime":
                total = total_reps_lifetime(user)
                th = info["threshold"]
                pct = min(100, int(total * 100 / th)) if th else 0
                st.progress(min(pct,100))
                st.write(f"{total}/{th} total reps")
            elif info["type"] == "session":
                best = best_session_reps(user)
                th = info["threshold"]
                pct = min(100, int(best * 100 / th)) if th else 0
                st.progress(min(pct,100))
                st.write(f"Best session: {best}/{th}")
            else:
                cnt = workouts_in_last_days(user, days=7)
                th = info["threshold"]
                pct = min(100, int(cnt * 100 / th)) if th else 0
                st.progress(min(pct,100))
                st.write(f"{cnt}/{th} days")

    if not badges.get("unlocked"):
        st.info("You haven't unlocked any badges yet. Do a workout session to start earning achievements!")

def page_landing():
    # Hero landing matching the provided screenshot
    # Prefer image hero first (explicit request), then fall back to video
    bg_video_url = None
    bg_video_uri = None

    # 1) Try image candidates first (prefer JPG)
    image_candidates = [
        "images/hero.jpg",
        "hero.jpg",
        "images/hero.jpeg",
        "images/hero.png",
    ]
    bg_uri = None
    for ip in image_candidates:
        bg_uri = _image_to_data_uri(ip)
        if bg_uri:
            break

    # 2) If no image found, try video candidates
    if not bg_uri:
        # Prefer Streamlit static folder for large video (served at /static/...)
        static_video_path = os.path.join("static", "hero.mp4")
        if os.path.exists(static_video_path):
            bg_video_url = "/static/hero.mp4"
        else:
            # Fallbacks: try to inline as data URI only if reasonably small
            video_candidates = [
                "videos/hero.mp4",
                "hero.mp4",
                "images/hero.mp4",
            ]
            for vp in video_candidates:
                if os.path.exists(vp):
                    try:
                        if os.path.getsize(vp) <= 15 * 1024 * 1024:  # 15 MB limit for data URI
                            bg_video_uri = _video_to_data_uri(vp)
                        else:
                            st.info("Hero video is large. For best performance, place it at static/hero.mp4 to stream directly.")
                    except Exception:
                        pass
                    break

    if not (bg_video_url or bg_video_uri or bg_uri):
        st.warning("No hero media found (images/hero.jpg or videos/hero.mp4). Showing placeholder background.")
 
    st.markdown(
        f"""
        <style>
        .hero-outer{{
            width: 100vw;
            margin-left: 50%;
            transform: translateX(-50%);
        }}
        .hero-wrap {{
            position: relative;
            height: 100vh;
            min-height: 680px;
            border-radius: 0;
            overflow: hidden;
            border: none;
            box-shadow: none;
            background: #0b0f17;
        }}
        .hero-img {{ position:absolute; inset:0; width:100%; height:100%; object-fit:cover; image-rendering:auto; }}
        .hero-overlay {{
            position:absolute; inset:0;
            background: linear-gradient(90deg, rgba(0,0,0,.85) 0%, rgba(0,0,0,.55) 40%, rgba(0,0,0,0) 70%);
        }}
        .hero-content {{
            position: relative; z-index: 2; height:100%; display:flex; flex-direction:column;
            justify-content:center; padding: 0 28px; max-width: 720px;
        }}
        .kicker {{ color:#e5e7eb; letter-spacing: .22em; font-weight:700; font-size:13px; opacity:.9; margin-bottom: 8px; }}
        .title {{ color:#ffffff; font-weight:900; font-size:64px; line-height:1; letter-spacing:.06em; text-transform:uppercase; }}
        .subtitle {{ color:#cbd5e1; margin-top:14px; font-size:16px; }}
        .cta-row {{ margin-top: 26px; display:flex; gap:14px; }}
        .cta {{ display:inline-block; padding: 12px 18px; border-radius: 8px; font-weight:800; letter-spacing:.04em; text-decoration:none; }}
        .cta.primary {{ background:#2563eb; color:#fff; }}
        .cta.primary:hover {{ background:#1e40af; }}
        .cta.ghost {{ color:#ffffff; background: rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.18); }}
        .cta.ghost:hover {{ background: rgba(255,255,255,0.2); }}
        @media (max-width: 900px){{
          .hero-wrap{{height:90vh; min-height:520px}}
          .title{{font-size:44px}}
          .subtitle{{font-size:14px}}
          .cta-row{{flex-wrap:wrap}}
        }}
        @media (max-width: 600px){{
          .hero-wrap{{height:85vh; min-height:460px}}
          .title{{font-size:34px}}
          .subtitle{{font-size:13px}}
        }}
        </style>
        <div class="hero-outer">
        <div class="hero-wrap"> 
          {f'<video class="hero-video" src="{bg_video_url}" autoplay muted loop playsinline></video>' if bg_video_url else (f'<video class="hero-video" src="{bg_video_uri}" autoplay muted loop playsinline></video>' if bg_video_uri else (f'<img class="hero-img" src="{bg_uri}">' if bg_uri else ''))}
          <div class="hero-overlay"></div>
          <div class="hero-content"> 
            <div class="kicker">THE SMART GYM TRAINER</div>
            <div class="title">ARE YOU<br>FIT-READY?</div>
            <div class="subtitle">AI-Powered Fitness Tracker</div>
            <div class="subtitle">Real-Time Reps • Posture Correction • Progress Logs</div>
            <div class="cta-row"> 
              <a class="cta primary" href="?page=Register">REGISTER NOW</a>
              <a class="cta ghost" href="?page=Login">LOGIN</a>
            </div>
          </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _image_to_data_uri(path: str):
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
                ext = os.path.splitext(path)[1].lower()
                mime = "image/png" if ext == ".png" else "image/jpeg"
                return f"data:{mime};base64,{b64}"
    except Exception:
        pass
    return None

def page_reports():
    st.title("📂 Reports / Sessions")
    user = st.session_state.get("auth_user")
    ensure_user_files(user)
    sp = get_user_sessions_path(user)
    if os.path.exists(sp):
        with open(sp, "r", encoding="utf-8") as f:
            contents = f.read()
            st.download_button("⬇ Download sessions CSV", contents, file_name=os.path.basename(sp), mime="text/csv")

        rows = sessions_list(user)
        agg = {m: 0 for m in exercise_modes}
        for r in rows:
            per = {}
            try:
                per = json.loads(r.get("per_ex_json") or "{}")
            except Exception:
                try:
                    per = eval(r.get("per_ex_json") or "{}")
                except Exception:
                    per = {}
            for k,v in per.items():
                try:
                    if k in agg:
                        agg[k] += int(v)
                except Exception:
                    pass

        EX_META = {
            "squat": ("🏋", "#0ea5a4"),
            "pushup": ("💪", "#f97316"),
            "jumping_jack": ("🤸", "#8b5cf6"),
            "high_knees": ("🦵", "#06b6d4"),
            "arm_raise": ("🙆", "#10b981"),
        }

        st.markdown("<div class='grid'>", unsafe_allow_html=True)
        for ex in exercise_modes:
            emoji, color = EX_META.get(ex, ("🏃", "#64748b"))
            val = int(agg.get(ex, 0))
            card_html = f"""
            <div class='col-3'>
              <div class='card' style='background:{color}; color:white;'>
                <div class='title-xs'>{emoji} {ex.replace('_',' ').title()}</div>
                <div class='value-xl'>{val}</div>
                <div class='subtext'>total reps</div>
              </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Charts: calories + reps over time
        rows_df = sessions_list(user)
        if rows_df:
            data = []
            for r in rows_df:
                try:
                    ts = pd.to_datetime(r.get("start_ts"))
                    total = int(r.get("total_reps", 0))
                    cals = int(r.get("calories", 0))
                    data.append({"date": ts, "total_reps": total, "calories": cals})
                except Exception:
                    pass
            if data:
                df = pd.DataFrame(data).sort_values("date")
                base = alt.Chart(df).encode(x=alt.X("date:T", title="Date"))
                reps_line = base.mark_line(color="#f43f5e").encode(y=alt.Y("total_reps:Q", title="Reps"))
                cal_bar = base.mark_bar(color="#22c55e", opacity=0.5).encode(y=alt.Y("calories:Q", title="Calories"))
                st.altair_chart(
                    alt.layer(cal_bar, reps_line).resolve_scale(y="independent").properties(height=240),
                    use_container_width=True
                )

        if st.button("🧹 Clear my sessions (reset)"):
            with open(sp, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["start_ts","end_ts","duration_s","total_reps","per_ex_json","calories"])
            st.success("Sessions cleared.")
            st.rerun()
    else:
        st.info("No sessions yet.")

# -------------------- Auth UI --------------------

def validate_user(user_or_email, password):
    ok, username = authenticate_user(user_or_email, password)
    return ok and bool(username)

def show_login():
    st.header("Welcome back")
    with st.form("login_form", clear_on_submit=False):
        user_or_email = st.text_input("Username or Email")
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In")
        if submitted:
            ok, username = authenticate_user(user_or_email, pw)
            if ok:
                st.session_state["auth_user"] = username
                st.session_state["page"] = "Home"
                ensure_user_files(username)
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials.")
    st.info("No account? Use Register in the sidebar.")

def show_register():
    st.header("Create your account")
    with st.form("register_form", clear_on_submit=False):
        colA, colB = st.columns(2)
        with colA:
            username = st.text_input("Username")
        with colB:
            email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        pw2 = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register")
        if submitted:
            if pw != pw2:
                st.error("Passwords do not match.")
            else:
                ok, msg = create_user(username, email, pw)
                if ok:
                    st.success(msg)
                    st.info("Go to Login to sign in.")
                else:
                    st.error(msg)


def main():
    auth_user = st.session_state.get("auth_user")
 
    if not auth_user:
        # Route via query params for clean button links from the landing hero
        qp = st.query_params
        page = (qp.get("page") or st.session_state.get("page") or "Landing")
        if isinstance(page, list):
            page = page[0]
        if page == "Login":
            st.session_state["page"] = "Login"
            show_login()
            return
        if page == "Register":
            st.session_state["page"] = "Register"
            show_register()
            return
        # Default: Landing page
        page_landing()
        return
 
    # logged in
    show_topbar()
    selected = show_navbar_sidebar()
    st.session_state["page"] = selected

    if selected == "Home":
        apply_dashboard_background_for_home()
        page_home()
    elif selected == "Workout":
        page_workout()
    elif selected == "Achievements":
        page_achievements()
    elif selected == "Reports":
        page_reports()
    elif selected == "Logout":
        for k in list(st.session_state.keys()):
            if k not in ("_rerun_count",):
                del st.session_state[k]
        st.rerun()

if __name__ == "__main__":
    main()
