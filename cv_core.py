"""
cv_core.py — Shared computer vision logic for Mimi AI Caregiver
Uses OpenCV Haar Cascade + NumPy only. No ML models, no internet.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = None


def _get_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    return _face_cascade


def _pixel_stats(gray):
    brightness = int(np.mean(gray))
    contrast   = int(np.std(gray))
    h, w       = gray.shape
    left       = gray[:, : w // 2]
    right      = gray[:, w // 2 :]
    right_flip = np.fliplr(right[:, : w // 2])
    min_w      = min(left.shape[1], right_flip.shape[1])
    symmetry   = int(np.mean(np.abs(left[:, :min_w].astype(int) - right_flip[:, :min_w].astype(int))))
    return brightness, contrast, symmetry


def _observations(brightness, contrast, symmetry, face_count):
    obs = []
    if brightness < 80:
        obs.append("Low brightness — environment may be too dark")
    elif brightness > 200:
        obs.append("High brightness — may cause sensory discomfort")
    else:
        obs.append("Lighting appears adequate")
    if contrast < 20:
        obs.append("Low contrast — image may be blurry or overexposed")
    elif contrast > 80:
        obs.append("High contrast — strong shadows present")
    if symmetry > 30:
        obs.append("Notable facial asymmetry detected — possible distress posture")
    else:
        obs.append("Facial symmetry within normal range")
    if face_count > 1:
        obs.append(f"{face_count} faces detected — multiple people in frame")
    return obs


def analyse_photo(pil_img):
    """
    pil_img: PIL Image (RGB)
    Returns: (results_dict, annotated_pil_img)
    """
    img_rgb  = np.array(pil_img.convert("RGB"))
    gray     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cascade  = _get_cascade()
    faces    = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    brightness, contrast, symmetry = _pixel_stats(gray)
    face_count    = len(faces) if hasattr(faces, "__len__") else 0
    face_detected = face_count > 0

    # cv_score: simple stress proxy (0-5)
    cv_score = 0
    if brightness < 60 or brightness > 210:
        cv_score += 1
    if contrast > 70:
        cv_score += 1
    if symmetry > 30:
        cv_score += 2
    if face_count == 0:
        cv_score += 1

    obs = _observations(brightness, contrast, symmetry, face_count)

    # Annotate
    annotated = pil_img.copy().convert("RGB")
    draw      = ImageDraw.Draw(annotated)
    if face_detected:
        for (x, y, w, h) in faces:
            draw.rectangle([x, y, x + w, y + h], outline=(50, 200, 100), width=3)

    results = {
        "face_detected": face_detected,
        "face_count":    face_count,
        "brightness":    brightness,
        "contrast":      contrast,
        "symmetry_score": symmetry,
        "observations":  obs,
        "cv_score":      cv_score,
    }
    return results, annotated


def analyse_video(video_path, sample_every=15, max_frames=60):
    """
    video_path: str path to video file
    Returns: (summary_dict, frame_stats_list, sample_pil_frames_list)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"observations": ["Could not open video file"], "avg_brightness": None,
                "avg_contrast": None, "avg_symmetry": None, "frames_with_face": 0,
                "frames_sampled": 0, "cv_score": 0}, [], []

    cascade      = _get_cascade()
    frame_idx    = 0
    sampled      = 0
    frame_stats  = []
    sample_pils  = []
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25

    while sampled < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces  = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            b, c, s = _pixel_stats(gray)
            has_face = len(faces) > 0 if hasattr(faces, "__len__") else False

            if has_face:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 200, 100), 2)

            frame_stats.append({
                "frame":      frame_idx,
                "time_s":     round(frame_idx / fps, 2),
                "brightness": b if has_face else None,
                "contrast":   c if has_face else None,
                "symmetry":   s if has_face else None,
                "face":       has_face,
            })

            if len(sample_pils) < 9:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sample_pils.append(Image.fromarray(rgb))

            sampled += 1
        frame_idx += 1

    cap.release()

    face_frames = [f for f in frame_stats if f["face"]]
    n_face      = len(face_frames)

    avg_b = int(np.mean([f["brightness"] for f in face_frames])) if n_face else None
    avg_c = int(np.mean([f["contrast"]   for f in face_frames])) if n_face else None
    avg_s = int(np.mean([f["symmetry"]   for f in face_frames])) if n_face else None

    obs = []
    if sampled == 0:
        obs.append("No frames could be read from video")
    elif n_face == 0:
        obs.append("No face detected in any sampled frame")
    else:
        obs.append(f"Face detected in {n_face}/{sampled} sampled frames")
        if avg_b is not None:
            if avg_b < 80:
                obs.append("Generally dim lighting across video")
            elif avg_b > 200:
                obs.append("Bright lighting — check for sensory glare")
            else:
                obs.append("Lighting consistent and adequate")
        if avg_s is not None and avg_s > 30:
            obs.append("Facial asymmetry noted across frames — possible distress")

    cv_score = 0
    if avg_b is not None and (avg_b < 60 or avg_b > 210):
        cv_score += 1
    if avg_s is not None and avg_s > 30:
        cv_score += 2
    if n_face == 0:
        cv_score += 1

    summary = {
        "observations":   obs,
        "avg_brightness": avg_b,
        "avg_contrast":   avg_c,
        "avg_symmetry":   avg_s,
        "frames_with_face": n_face,
        "frames_sampled": sampled,
        "cv_score":       cv_score,
    }
    return summary, frame_stats, sample_pils
