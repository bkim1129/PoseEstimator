"""
MediaPipe Pose Overlay App (Streamlit)
- Upload image or video
- Run MediaPipe Pose (single person) or MediaPipe Tasks Pose Landmarker (multi-person)
- Overlay skeleton and selected joint angles (unsigned or signed)
- Save processed image/video and download

How to run:
1) Create a venv and install deps:
   pip install streamlit mediapipe opencv-python numpy pillow
   # For Apple Silicon: prefer Python 3.11/3.12 (arm64) if possible; Python 3.9 also works.
2) Start app:
   streamlit run mediapipe_pose_app.py

Notes:
- Outputs are written to ./outputs
- Multi-person uses an auto-downloaded .task model (cached under ~/.cache/mediapipe_tasks)
"""

from __future__ import annotations

import os
import io
import math
import time
import tempfile
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("mediapipe not installed. Run: pip install mediapipe")

# =============================
# UI THEME
# =============================
PRIMARY = "#0ea5e9"   # sky blue
SECONDARY = "#0284c7" # deep blue
GRAY_900 = "#0f172a"
RADIUS_PX = 12

st.set_page_config(page_title="Pose Overlay", page_icon="🧍", layout="wide")

CUSTOM_CSS = f"""
<style>
header[data-testid="stHeader"] {{
  background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
  height: 64px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}}
.block-container {{ padding-top: 1.2rem; }}

/* Soft, rounded controls */
div.stButton>button, .stDownloadButton button, .stFileUploader label {{
  border-radius: {RADIUS_PX}px !important; box-shadow: 0 4px 6px rgba(0,0,0,0.10);
  transition: transform 0.2s ease;
}}
div.stButton>button:hover, .stDownloadButton button:hover {{ transform: scale(1.02); }}

.card {{ background: white; border-radius: {RADIUS_PX}px; box-shadow: 0 4px 6px rgba(0,0,0,0.10); padding: 1rem 1.25rem; margin-bottom: 1rem; }}
.h-title {{ font-weight: 700; color: {GRAY_900}; border-left: 6px solid {PRIMARY}; padding-left: 10px; margin-bottom: 0.5rem; }}
section[data-testid="stSidebar"] > div {{ width: 290px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================
# MediaPipe setup (solutions + tasks)
# =============================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
from mediapipe.framework.formats import landmark_pb2
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
L = mp_pose.PoseLandmark

# Try MediaPipe Tasks (for multi-person)
try:
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
    from mediapipe.tasks.python.core.base_options import BaseOptions
    TASKS_AVAILABLE = True
except Exception:
    PoseLandmarker = None
    PoseLandmarkerOptions = None
    RunningMode = None
    BaseOptions = None
    TASKS_AVAILABLE = False

POSE_TASK_URL_FULL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

def ensure_pose_task_model(url: str = POSE_TASK_URL_FULL, cache_dir: Optional[str] = None) -> str:
    """Ensure the Pose Landmarker .task file exists locally; download if missing."""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe_tasks")
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(dest) or os.path.getsize(dest) < 1024 * 1024:
        import urllib.request
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception:
            local_dir = os.path.join(os.getcwd(), "models"); os.makedirs(local_dir, exist_ok=True)
            dest = os.path.join(local_dir, os.path.basename(url))
            urllib.request.urlretrieve(url, dest)
    return dest

# =============================
# Angle definitions (A, B, C) => angle at B
# =============================
ANGLE_JOINTS: Dict[str, Tuple[int, int, int]] = {
    "Right Elbow": (L.RIGHT_SHOULDER.value, L.RIGHT_ELBOW.value, L.RIGHT_WRIST.value),
    "Left Elbow": (L.LEFT_SHOULDER.value, L.LEFT_ELBOW.value, L.LEFT_WRIST.value),
    "Right Shoulder": (L.RIGHT_ELBOW.value, L.RIGHT_SHOULDER.value, L.RIGHT_HIP.value),
    "Left Shoulder": (L.LEFT_ELBOW.value, L.LEFT_SHOULDER.value, L.LEFT_HIP.value),
    "Right Hip": (L.RIGHT_SHOULDER.value, L.RIGHT_HIP.value, L.RIGHT_KNEE.value),
    "Left Hip": (L.LEFT_SHOULDER.value, L.LEFT_HIP.value, L.LEFT_KNEE.value),
    "Right Knee": (L.RIGHT_HIP.value, L.RIGHT_KNEE.value, L.RIGHT_ANKLE.value),
    "Left Knee": (L.LEFT_HIP.value, L.LEFT_KNEE.value, L.LEFT_ANKLE.value),
    # Use toes (FOOT_INDEX) for ankle instead of heel for a stable foot vector
    "Right Ankle": (L.RIGHT_KNEE.value, L.RIGHT_ANKLE.value, L.RIGHT_FOOT_INDEX.value),
    "Left Ankle": (L.LEFT_KNEE.value, L.LEFT_ANKLE.value, L.LEFT_FOOT_INDEX.value),
}

# =============================
# Options & math
# =============================
@dataclass
class PoseOptions:
    show_skeleton: bool = True
    show_angles: bool = True
    selected_joints: Tuple[str, ...] = tuple()
    visibility_thresh: float = 0.5
    thickness: int = 2
    circle_radius: int = 3
    model_complexity: int = 1
    smooth_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    angle_mode: str = "unsigned"  # "unsigned" or "signed"
    # Multi-person options
    engine: str = "Single person (fast)"  # or "Multi person (choose person)"
    num_poses: int = 5
    target_person_index: int = 1  # 1-based in UI
    model_path: Optional[str] = None
    person_order: str = "Largest area"  # selection ordering for indexing
    # Tracking stability
    tracking_mode: str = "Stable (strict gating)"  # Stable | Adaptive | Loose
    gate: float = 0.35
    ema_alpha: float = 0.6
    max_lost_frames: int = 15


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b; bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return float("nan")
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def _signed_angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Signed angle ABC (degrees) in screen coords (y down), CCW positive."""
    ba = a - b; bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return float("nan")
    dot = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    ang = math.degrees(math.acos(dot))
    cross_z = ba[0] * bc[1] - ba[1] * bc[0]
    if cross_z < 0:
        ang = -ang
    return ang


def _landmark_xy(landmarks, idx: int, w: int, h: int) -> Optional[Tuple[int, int, float]]:
    lm = landmarks[idx]
    vis = float(getattr(lm, "visibility", 1.0) or 1.0)
    return int(lm.x * w), int(lm.y * h), vis


def draw_pose_and_angles(image_bgr: np.ndarray, results, opts: PoseOptions) -> Tuple[np.ndarray, Dict[str, float]]:
    out = image_bgr.copy(); angle_values: Dict[str, float] = {}
    if results.pose_landmarks:
        if opts.show_skeleton:
            mp_drawing.draw_landmarks(
                out, results.pose_landmarks, POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        if opts.show_angles and opts.selected_joints:
            h, w = out.shape[:2]; lms = results.pose_landmarks.landmark
            for joint_name in opts.selected_joints:
                tri = ANGLE_JOINTS.get(joint_name)
                if not tri: continue
                a_idx, b_idx, c_idx = tri
                a = _landmark_xy(lms, a_idx, w, h); b = _landmark_xy(lms, b_idx, w, h); c = _landmark_xy(lms, c_idx, w, h)
                if not a or not b or not c: continue
                ax, ay, av = a; bx, by, bv = b; cx, cy, cv = c
                if (av < opts.visibility_thresh) or (bv < opts.visibility_thresh) or (cv < opts.visibility_thresh):
                    continue
                ang = (_signed_angle_between(np.array([ax, ay], float), np.array([bx, by], float), np.array([cx, cy], float))
                       if opts.angle_mode == "signed"
                       else _angle_between(np.array([ax, ay], float), np.array([bx, by], float), np.array([cx, cy], float)))
                if math.isnan(ang):
                    continue
                angle_values[joint_name] = ang
                # Green label with black outline for visibility
                cv2.circle(out, (bx, by), max(3, opts.circle_radius), (0, 255, 0), -1)
                # ASCII-only label to avoid '?' with OpenCV Hershey fonts
                val_text = f"{ang:+.1f}" if opts.angle_mode == "signed" else f"{ang:.1f}"
                label = f"{joint_name}: {val_text} deg"
                cv2.putText(out, label, (bx + 8, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(out, label, (bx + 8, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return out, angle_values

# =============================
# Single-person processing (solutions.pose)
# =============================

def process_image_bytes(img_bytes: bytes, opts: PoseOptions) -> Tuple[np.ndarray, Dict[str, float]]:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=opts.model_complexity,
        smooth_landmarks=opts.smooth_landmarks,
        min_detection_confidence=opts.min_detection_confidence,
        min_tracking_confidence=opts.min_tracking_confidence,
    ) as pose:
        results = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return draw_pose_and_angles(img_bgr, results, opts)


def process_video_file(input_path: str, output_path: str, opts: PoseOptions, scale: float = 1.0, progress_cb=None) -> Dict[str, float]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = int(w * scale), int(h * scale)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    last_angles: Dict[str, float] = {}
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=opts.model_complexity,
        smooth_landmarks=opts.smooth_landmarks,
        min_detection_confidence=opts.min_detection_confidence,
        min_tracking_confidence=opts.min_tracking_confidence,
    ) as pose:
        frame_idx = 0; total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        while True:
            ok, frame = cap.read();
            if not ok: break
            frame_idx += 1
            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            drawn, angles = draw_pose_and_angles(frame, results, opts)
            last_angles.update({k: v for k, v in angles.items()})
            writer.write(drawn)
            if progress_cb and total > 0:
                progress_cb(min(frame_idx / total, 1.0))
    cap.release(); writer.release()
    return last_angles

# =============================
# Multi-person processing (Tasks Pose Landmarker)
# =============================

def _bbox_center(landmarks, w: int, h: int) -> Tuple[float, float, float, int]:
    xs = [lm.x for lm in landmarks]; ys = [lm.y for lm in landmarks]
    minx, maxx = max(min(xs), 0.0), min(max(xs), 1.0)
    miny, maxy = max(min(ys), 0.0), min(max(ys), 1.0)
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    area = (maxx - minx) * (maxy - miny)
    return cx * w, cy * h, float(min(area, 1.0)), len(xs)


def _sort_candidates(candidates, w: int, h: int, order: str):
    centers = []
    for lms in candidates:
        cx, cy, area, _ = _bbox_center(lms, w, h)
        centers.append((lms, cx, cy, area))
    if order == "Largest area":
        centers.sort(key=lambda t: t[3], reverse=True)
    elif order == "Left-to-right":
        centers.sort(key=lambda t: t[1])
    elif order == "Right-to-left":
        centers.sort(key=lambda t: t[1], reverse=True)
    elif order == "Top-to-bottom":
        centers.sort(key=lambda t: t[2])
    elif order == "Bottom-to-top":
        centers.sort(key=lambda t: t[2], reverse=True)
    return centers


def _wrap_result_from_lms(lms_list) -> object:
    """Convert a Tasks landmark list into a proto NormalizedLandmarkList so
    drawing_utils.draw_landmarks() can operate without AttributeError(HasField).
    """
    lmlist = landmark_pb2.NormalizedLandmarkList()
    for lm in lms_list:
        n = landmark_pb2.NormalizedLandmark()
        n.x = float(getattr(lm, 'x', 0.0))
        n.y = float(getattr(lm, 'y', 0.0))
        if hasattr(lm, 'z'):
            n.z = float(getattr(lm, 'z', 0.0))
        if hasattr(lm, 'visibility'):
            try:
                n.visibility = float(lm.visibility)
            except Exception:
                pass
        lmlist.landmark.append(n)

    class _R:
        def __init__(self, pl):
            self.pose_landmarks = pl
    return _R(lmlist)

# --- Tracking helpers ---

def _landmarks_to_np(lms_list, w: int, h: int) -> np.ndarray:
    """Return Nx2 array of pixel coords for landmarks."""
    return np.array([[lm.x * w, lm.y * h] for lm in lms_list], dtype=float)


def _normalize_pts(pts: np.ndarray) -> np.ndarray:
    centered = pts - pts.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(centered, axis=1).mean() + 1e-6
    return centered / scale


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1); b = b.reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
    return float(np.dot(a, b) / denom)


def _tracking_cost(prev_np: np.ndarray, pred_center: np.ndarray, prev_area: float,
                   cand_lms, w: int, h: int,
                   w_center: float = 0.45, w_shape: float = 0.45, w_area: float = 0.10) -> float:
    """Lower is better: combines center prediction, shape, and area differences."""
    cand_np = _landmarks_to_np(cand_lms, w, h)
    cand_center = cand_np.mean(axis=0)
    center_dist = np.linalg.norm(cand_center - pred_center) / (math.hypot(w, h) + 1e-6)
    prev_n = _normalize_pts(prev_np)
    cand_n = _normalize_pts(cand_np)
    shape_dist = 1.0 - _cosine(prev_n, cand_n)
    _, _, area_cand, _ = _bbox_center(cand_lms, w, h)
    area_diff = abs(area_cand - prev_area)
    return w_center * center_dist + w_shape * shape_dist + w_area * area_diff

# --- ROI helpers for hybrid pipeline ---

def _clip_box(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[int, int, int, int]:
    w = max(4.0, min(w, float(W)))
    h = max(4.0, min(h, float(H)))
    x = min(max(0.0, x), float(W) - w)
    y = min(max(0.0, y), float(H) - h)
    return int(x), int(y), int(w), int(h)


def _roi_from_lms(lms_list, W: int, H: int, pad: float = 0.35) -> Tuple[int, int, int, int]:
    xs = [lm.x * W for lm in lms_list]; ys = [lm.y * H for lm in lms_list]
    minx, maxx = max(min(xs), 0.0), min(max(xs), float(W))
    miny, maxy = max(min(ys), 0.0), min(max(ys), float(H))
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    bw, bh = (maxx - minx), (maxy - miny)
    # Ensure reasonable size even if landmarks collapse
    bw = max(bw, 0.15 * W); bh = max(bh, 0.25 * H)
    bw *= (1.0 + 2.0 * pad); bh *= (1.0 + 2.0 * pad)
    x = cx - bw / 2.0; y = cy - bh / 2.0
    return _clip_box(x, y, bw, bh, W, H)


def _wrap_solution_to_full(result_pose, roi_box: Tuple[int, int, int, int], W: int, H: int):
    if not result_pose or not result_pose.pose_landmarks:
        return None
    x0, y0, rw, rh = roi_box
    lmlist = landmark_pb2.NormalizedLandmarkList()
    for lm in result_pose.pose_landmarks.landmark:
        n = landmark_pb2.NormalizedLandmark()
        fx = (x0 + lm.x * rw) / float(W)
        fy = (y0 + lm.y * rh) / float(H)
        n.x = float(np.clip(fx, 0.0, 1.0))
        n.y = float(np.clip(fy, 0.0, 1.0))
        if hasattr(lm, 'z'):
            n.z = float(lm.z)
        if hasattr(lm, 'visibility'):
            try:
                n.visibility = float(lm.visibility)
            except Exception:
                pass
        lmlist.landmark.append(n)
    class _R:
        def __init__(self, pl): self.pose_landmarks = pl
    return _R(lmlist)


def _create_landmarker(model_path: str, running_mode, num_poses: int = 5) -> Optional[object]:
    if not TASKS_AVAILABLE or not model_path:
        return None
    try:
        base = BaseOptions(model_asset_path=model_path)
        options = PoseLandmarkerOptions(base_options=base, running_mode=running_mode, num_poses=num_poses)
        return PoseLandmarker.create_from_options(options)
    except Exception:
        return None


def process_image_bytes_tasks(img_bytes: bytes, opts: PoseOptions, model_path: Optional[str]) -> Tuple[np.ndarray, Dict[str, float]]:
    if not TASKS_AVAILABLE:
        raise RuntimeError("MediaPipe Tasks unavailable.")
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    model_path = model_path or ensure_pose_task_model()
    landmarker = _create_landmarker(model_path, RunningMode.IMAGE, opts.num_poses)
    if landmarker is None:
        raise RuntimeError("Failed to initialize PoseLandmarker. Check model file.")
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return img_bgr, {}
    h, w = img_bgr.shape[:2]
    ordered = _sort_candidates(result.pose_landmarks, w, h, opts.person_order)
    idx = max(0, min(opts.target_person_index - 1, len(ordered) - 1))
    chosen = ordered[idx][0]
    wrapped = _wrap_result_from_lms(chosen)
    return draw_pose_and_angles(img_bgr, wrapped, opts)


def process_video_file_tasks(input_path: str, output_path: str, opts: PoseOptions, model_path: Optional[str], scale: float = 1.0, progress_cb=None) -> Dict[str, float]:
    """
    Hybrid pipeline for multi-person video:
      1) Use Tasks only on the first frame to choose the target by ordering+index
      2) Switch to the same solver as single-person (solutions.Pose) on a cropped ROI
         around the chosen person for all subsequent frames.
    This honors your request to "use the same pose estimation algorithm" after selection.
    """
    if not TASKS_AVAILABLE:
        raise RuntimeError("MediaPipe Tasks unavailable.")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_W, out_H = int(W * scale), int(H * scale)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_W, out_H))

    model_path = model_path or ensure_pose_task_model()
    landmarker = _create_landmarker(model_path, RunningMode.VIDEO, opts.num_poses)
    if landmarker is None:
        raise RuntimeError("Failed to initialize PoseLandmarker. Check model file.")

    roi_box: Optional[Tuple[int, int, int, int]] = None
    lost = 0
    last_angles: Dict[str, float] = {}
    frame_idx = 0; total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Create single-person solver once (same as single-person option)
    pose_solver = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=opts.model_complexity,
        smooth_landmarks=opts.smooth_landmarks,
        min_detection_confidence=opts.min_detection_confidence,
        min_tracking_confidence=opts.min_tracking_confidence,
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # Downscale output frame if requested (processing on original for accuracy)
            draw_frame = frame.copy()

            if roi_box is None:
                # FIRST FRAME: choose target with Tasks and compute ROI
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ts_ms = int((frame_idx / fps) * 1000.0)
                result = landmarker.detect_for_video(mp_image, ts_ms)
                if result.pose_landmarks:
                    ordered = _sort_candidates(result.pose_landmarks, W, H, opts.person_order)
                    idx = max(0, min(opts.target_person_index - 1, len(ordered) - 1))
                    chosen_lms = ordered[idx][0]
                    # ROI from chosen landmarks, then immediately run solutions.Pose for consistency
                    roi_box = _roi_from_lms(chosen_lms, W, H, pad=0.35)
                else:
                    # Nothing detected yet; just write original frame
                    writer.write(cv2.resize(draw_frame, (out_W, out_H)) if scale != 1.0 else draw_frame)
                    if progress_cb and total > 0:
                        progress_cb(min(frame_idx / total, 1.0))
                    continue

            # Run single-person solver on ROI
            x0, y0, rw, rh = roi_box
            roi = frame[y0:y0+rh, x0:x0+rw]
            if roi.size == 0:
                lost += 1
                if lost > opts.max_lost_frames:
                    roi_box = None
                writer.write(cv2.resize(draw_frame, (out_W, out_H)) if scale != 1.0 else draw_frame)
                if progress_cb and total > 0:
                    progress_cb(min(frame_idx / total, 1.0))
                continue

            res = pose_solver.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            wrapped = _wrap_solution_to_full(res, roi_box, W, H)

            if wrapped is None:
                # Expand ROI a bit and count as lost
                lost += 1
                # Adaptive expansion
                exp = min(1.8, 1.15 + 0.05 * lost)
                cx = x0 + rw / 2.0; cy = y0 + rh / 2.0
                new_w = min(W, rw * exp); new_h = min(H, rh * exp)
                roi_box = _clip_box(cx - new_w/2.0, cy - new_h/2.0, new_w, new_h, W, H)
                drawn = draw_frame
            else:
                # Draw and compute angles on full frame coordinates
                drawn, angles = draw_pose_and_angles(draw_frame, wrapped, opts)
                last_angles.update({k: v for k, v in angles.items()})

                # Update ROI from new landmarks with smoothing
                lms = wrapped.pose_landmarks.landmark
                nx0, ny0, nW, nH = _roi_from_lms(lms, W, H, pad=0.30)
                alpha = opts.ema_alpha
                sx0 = int(alpha * nx0 + (1 - alpha) * x0)
                sy0 = int(alpha * ny0 + (1 - alpha) * y0)
                sW = int(alpha * nW + (1 - alpha) * rw)
                sH = int(alpha * nH + (1 - alpha) * rh)
                roi_box = _clip_box(sx0, sy0, sW, sH, W, H)
                lost = 0

            # Write frame
            if scale != 1.0:
                drawn = cv2.resize(drawn, (out_W, out_H), interpolation=cv2.INTER_AREA)
            writer.write(drawn)
            if progress_cb and total > 0:
                progress_cb(min(frame_idx / total, 1.0))
    finally:
        pose_solver.close() if hasattr(pose_solver, 'close') else None
        cap.release(); writer.release()

    return last_angles

# =============================
# Sidebar (Controls)
# =============================
with st.sidebar:
    st.markdown("<div class='h-title'>Controls</div>", unsafe_allow_html=True)

    mode = st.radio("Input Type", ["Image", "Video"], index=0, help="Choose whether to process a single image or a video.")

    uploaded = st.file_uploader(
        "Upload file",
        type=["png", "jpg", "jpeg", "mp4", "mov", "avi", "mkv"],
        accept_multiple_files=False,
        help="PNG/JPG for Image, MP4/MOV/AVI/MKV for Video",
    )

    st.divider()

    # Engine selection (auto-download model for multi-person)
    engine = st.selectbox("Engine", ["Single person (fast)", "Multi person (choose person)"])
    num_poses = 5
    target_person = 1
    if engine == "Multi person (choose person)":
        st.caption("Using auto-downloaded pose_landmarker_full.task (cached).")
        num_poses = st.slider("Max people to detect", 1, 6, 5)
        target_person = st.number_input("Target person index (1-based)", min_value=1, max_value=6, value=1, step=1)
        person_order = st.selectbox(
            "Person ordering for index",
            ["Largest area", "Left-to-right", "Right-to-left", "Top-to-bottom", "Bottom-to-top"],
            index=0,
            help="Controls how index 1,2,3… maps to people when multiple are present.",
        )
        st.markdown("**Tracking**")
        tracking_mode = st.selectbox("Tracking mode", ["Stable (strict gating)", "Adaptive (reacquire under gate×2)", "Loose (always pick min-cost)"])
        gate = st.slider("Tracking gate (lower=stricter)", 0.1, 0.8, 0.35, 0.01)
        ema_alpha = st.slider("Smoothing (EMA α)", 0.0, 1.0, 0.6, 0.05)
        max_lost = st.slider("Max lost frames before pause", 1, 60, 15)
    else:
        person_order = "Largest area"
        tracking_mode = "Stable (strict gating)"
        gate = 0.35
        ema_alpha = 0.6
        max_lost = 15

    st.markdown("**Overlay")
    show_skeleton = st.checkbox("Show skeleton", value=True)
    show_angles = st.checkbox("Show joint angles", value=True)
    selected_joints = st.multiselect("Joints to compute", list(ANGLE_JOINTS.keys()), default=["Right Knee", "Left Knee"])  # sensible default

    angle_mode_label = st.selectbox("Angle output", ["Unsigned (0–180°)", "Signed (−180° to +180°)"], index=0)
    angle_mode = "signed" if "Signed" in angle_mode_label else "unsigned"

    thickness = st.slider("Line thickness", 1, 5, 2)
    circle_radius = st.slider("Point size", 1, 8, 3)
    visibility_thresh = st.slider("Visibility threshold", 0.0, 1.0, 0.5, 0.05)

    st.divider()

    st.markdown("**Model")
    model_complexity = st.select_slider("Model complexity", options=[0, 1, 2], value=1)
    smooth_landmarks = st.checkbox("Smooth landmarks", value=True)
    min_det_conf = st.slider("Min detection confidence", 0.0, 1.0, 0.5, 0.05)
    min_trk_conf = st.slider("Min tracking confidence", 0.0, 1.0, 0.5, 0.05)
    video_scale = st.slider("Video scale", 0.25, 1.0, 0.75, 0.05, help="Downscale for faster processing")

    with st.expander("Debug & Tests"):
        if st.button("Run sanity tests"):
            try:
                msgs = []
                # 1) Angle math sanity
                A = np.array([0.0, 0.0]); B = np.array([1.0, 0.0]); C = np.array([2.0, 0.0])
                ang = _angle_between(A, B, C)
                msgs.append(f"Straight line expected 180°, got {ang:.2f}°")
                A = np.array([0.0, 0.0]); B = np.array([1.0, 0.0]); C = np.array([1.0, 1.0])
                ang = _angle_between(A, B, C)
                msgs.append(f"Right angle expected 90°, got {ang:.2f}°")

                # 1b) Signed angle orientation checks
                ang_s = _signed_angle_between(np.array([0.0, -1.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0]))
                msgs.append(f"Signed +90° check (up->right), got {ang_s:.2f}°")
                ang_s2 = _signed_angle_between(np.array([0.0, -1.0]), np.array([0.0, 0.0]), np.array([-1.0, 0.0]))
                msgs.append(f"Signed -90° check (up->left), got {ang_s2:.2f}°")

                # 2) Ankle mapping sanity (Knee-Ankle-FootIndex colinear)
                h, w = 200, 300
                class _LM:  # minimal landmark
                    def __init__(self, x, y, v=1.0): self.x, self.y, self.visibility = x, y, v
                class _PL:  # wrapper
                    def __init__(self, lms): self.landmark = lms
                class _R:
                    def __init__(self, pl): self.pose_landmarks = pl
                max_idx = max([idx for triplet in ANGLE_JOINTS.values() for idx in triplet])
                lms = [_LM(0.0, 0.0, 0.0) for _ in range(max_idx + 1)]
                lms[L.RIGHT_KNEE.value] = _LM(0.40, 0.50, 1.0)
                lms[L.RIGHT_ANKLE.value] = _LM(0.50, 0.50, 1.0)
                lms[L.RIGHT_FOOT_INDEX.value] = _LM(0.60, 0.50, 1.0)
                res = _R(_PL(lms))
                test_opts = PoseOptions(show_skeleton=False, show_angles=True, selected_joints=("Right Ankle",), visibility_thresh=0.0)
                img = np.zeros((h, w, 3), dtype=np.uint8)
                _, angles = draw_pose_and_angles(img, res, test_opts)
                ankle = angles.get("Right Ankle", float('nan'))
                msgs.append(f"Right Ankle colinear ~180°, got {ankle:.2f}°")

                # 3) Tasks -> proto conversion sanity
                class _TL:
                    def __init__(self, xs, ys):
                        self.lms = [type('P', (), {'x': x, 'y': y}) for x, y in zip(xs, ys)]
                    def __iter__(self): return iter(self.lms)
                t_lms = _TL([0.40, 0.50, 0.60], [0.50, 0.50, 0.50])
                wrapped = _wrap_result_from_lms(list(t_lms))
                img2 = np.zeros((h, w, 3), dtype=np.uint8)
                _, angles2 = draw_pose_and_angles(img2, wrapped, test_opts)
                msgs.append("Tasks→proto conversion OK (no HasField error)")

                # 4) Ordering & indexing sanity
                two = [
                    [type('P', (), {'x':0.10,'y':0.10}), type('P', (), {'x':0.20,'y':0.20})],  # left
                    [type('P', (), {'x':0.80,'y':0.10}), type('P', (), {'x':0.90,'y':0.20})],  # right
                ]
                ordered_lr = _sort_candidates(two, 1000, 1000, "Left-to-right")
                ordered_rl = _sort_candidates(two, 1000, 1000, "Right-to-left")
                assert ordered_lr[0][1] < ordered_rl[0][1], "Ordering logic failed"
                msgs.append("Ordering tests passed")

                st.success("\n".join(msgs))
            except Exception as e:
                st.exception(e)

# Resolve model path (auto-download) if needed
_model_tmp_path: Optional[str] = None
if engine == "Multi person (choose person)":
    try:
        _model_tmp_path = ensure_pose_task_model()
    except Exception:
        _model_tmp_path = None

# Build options object (single source of truth)
opts = PoseOptions(
    show_skeleton=show_skeleton,
    show_angles=show_angles,
    selected_joints=tuple(selected_joints),
    visibility_thresh=visibility_thresh,
    thickness=thickness,
    circle_radius=circle_radius,
    model_complexity=model_complexity,
    smooth_landmarks=smooth_landmarks,
    min_detection_confidence=min_det_conf,
    min_tracking_confidence=min_trk_conf,
    angle_mode=angle_mode,
    engine=engine,
    num_poses=int(num_poses),
    target_person_index=int(target_person),
    model_path=_model_tmp_path,
    person_order=person_order,
    tracking_mode=tracking_mode,
    gate=float(gate),
    ema_alpha=float(ema_alpha),
    max_lost_frames=int(max_lost),
)

# =============================
# Main layout
# =============================
left, right = st.columns([7, 5])

with left:
    st.markdown("<div class='h-title'>Pose Estimation</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>Upload an image or video, choose joints, and export the overlay.</div>", unsafe_allow_html=True)

    if uploaded is None:
        st.info("Upload a file to begin.")
    else:
        file_bytes = uploaded.read()
        name = uploaded.name.lower()

        if mode == "Image" and any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
            st.write("**Preview**"); st.image(file_bytes, caption=uploaded.name, use_column_width=True)
            if st.button("Process Image", type="primary"):
                with st.spinner("Running pose on image…"):
                    if opts.engine == "Multi person (choose person)":
                        if not TASKS_AVAILABLE:
                            st.error("MediaPipe Tasks not available in this environment."); st.stop()
                        if not opts.model_path:
                            st.error("Could not auto-download the Pose Landmarker model. Check your internet connection."); st.stop()
                        im_out, angles = process_image_bytes_tasks(file_bytes, opts, opts.model_path)
                    else:
                        im_out, angles = process_image_bytes(file_bytes, opts)
                st.success("Done")
                st.image(cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB), caption="Overlay", use_column_width=True)
                os.makedirs("outputs", exist_ok=True)
                out_path = os.path.join("outputs", f"overlay_{int(time.time())}.png"); cv2.imwrite(out_path, im_out)
                buf = io.BytesIO(); Image.fromarray(cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
                st.download_button("Download PNG", data=buf.getvalue(), file_name=os.path.basename(out_path), mime="image/png")
                if angles:
                    st.json({k: round(v, 1) for k, v in angles.items()})

        elif mode == "Video" and any(name.endswith(ext) for ext in [".mp4", ".mov", ".avi", ".mkv"]):
            st.write("**Preview**"); st.video(file_bytes)
            if st.button("Process Video", type="primary"):
                with st.spinner("Running pose on video…"):
                    os.makedirs("outputs", exist_ok=True)
                    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1])
                    tmp_in.write(file_bytes); tmp_in.flush(); tmp_in.close()
                    out_path = os.path.join("outputs", f"overlay_{int(time.time())}.mp4")
                    prog = st.progress(0.0)
                    def _cb(p: float): prog.progress(p)
                    if opts.engine == "Multi person (choose person)":
                        if not TASKS_AVAILABLE:
                            st.error("MediaPipe Tasks not available in this environment."); st.stop()
                        if not opts.model_path:
                            st.error("Could not auto-download the Pose Landmarker model. Check your internet connection."); st.stop()
                        last_angles = process_video_file_tasks(tmp_in.name, out_path, opts, model_path=opts.model_path, scale=video_scale, progress_cb=_cb)
                    else:
                        last_angles = process_video_file(tmp_in.name, out_path, opts, scale=video_scale, progress_cb=_cb)
                st.success("Done"); st.video(out_path)
                with open(out_path, "rb") as f:
                    st.download_button("Download MP4", data=f.read(), file_name=os.path.basename(out_path), mime="video/mp4")
                if last_angles:
                    st.json({k: round(v, 1) for k, v in last_angles.items()})
        else:
            st.warning("Input type and file extension don't match. Switch the 'Input Type' or upload a matching file.")

with right:
    st.markdown("<div class='h-title'>Tips</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='card'>
        <ul>
          <li>Use <b>Video scale</b> to speed up long videos.</li>
          <li>Increase <b>visibility threshold</b> to filter uncertain landmarks.</li>
          <li>Angles are computed at the middle joint (e.g., Knee: Hip–Knee–Ankle).</li>
          <li>Try <b>model complexity 2</b> for precision; 0/1 for speed.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='h-title'>Session</div>", unsafe_allow_html=True)
    st.code(
        f"Selected joints: {', '.join(opts.selected_joints) if opts.selected_joints else 'None'}\n"
        f"Skeleton: {opts.show_skeleton} | Angles: {opts.show_angles}\n"
        f"Model complexity: {opts.model_complexity} | Smooth: {opts.smooth_landmarks}\n"
        f"Det conf: {opts.min_detection_confidence:.2f} | Trk conf: {opts.min_tracking_confidence:.2f}\n"
        f"Engine: {opts.engine} | Target person: {opts.target_person_index} | Max poses: {opts.num_poses}\n"
        f"Ordering: {opts.person_order} | Angle mode: {opts.angle_mode} | Tracking: {opts.tracking_mode} (gate={opts.gate:.2f}, α={opts.ema_alpha:.2f})"
    )

st.markdown(
    f"""
    <hr/>
    <div style='color:{GRAY_900}; opacity:0.8;'>
      <small>© Unity Move Physical Therapy Wellness PLLC • Built with MediaPipe + OpenCV + Streamlit</small>
    </div>
    """,
    unsafe_allow_html=True,
)
