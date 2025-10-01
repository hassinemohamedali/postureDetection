import sys
import streamlit as st
st.sidebar.info(f"🐍 Python version: {sys.version}")
"""
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Mediapipe pose & drawing
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


def check_guard(draw, wrist, shoulder, elbow, h, w):
    # Convert normalized landmarks to pixel coordinates
    wrist_xy = (int(wrist.x * w), int(wrist.y * h))
    elbow_xy = (int(elbow.x * w), int(elbow.y * h))
    shoulder_xy = (int(shoulder.x * w), int(shoulder.y * h))

    # Check if wrist is above shoulder
    if wrist.y < shoulder.y:
        color = (0, 255, 0)  # green
    else:
        color = (255, 0, 0)  # red

    # Pillow expects RGB tuple
    color_rgb = (color[0], color[1], color[2])

    # Draw custom lines (shoulder → elbow → wrist)
    draw.line([shoulder_xy, elbow_xy], fill=color_rgb, width=3)
    draw.line([elbow_xy, wrist_xy], fill=color_rgb, width=3)

    # Draw small circles
    r = 6
    for xy in [shoulder_xy, elbow_xy, wrist_xy]:
        draw.ellipse([xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r], fill=color_rgb)


class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert to numpy array (RGB for PIL)
        img = frame.to_ndarray(format="rgb24")
        h, w, _ = img.shape

        # Run pose detection
        results = self.pose.process(img)

        # Convert to PIL for drawing
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        if results.pose_landmarks:
            # Draw skeleton using mediapipe (requires numpy array)
            annotated = np.array(pil_img)
            mp_draw.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Convert back to PIL for custom drawing
            pil_img = Image.fromarray(annotated)
            draw = ImageDraw.Draw(pil_img)

            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

            # Apply guard checks
            check_guard(draw, left_wrist, left_shoulder, left_elbow, h, w)
            check_guard(draw, right_wrist, right_shoulder, right_elbow, h, w)
        else:
            # Add "No pose detected" text
            draw.text((10, 40), "No pose detected", fill=(255, 0, 0))

        # Convert back to numpy and return as frame
        return av.VideoFrame.from_ndarray(np.array(pil_img), format="rgb24")

    def __del__(self):
        try:
            self.pose.close()
        except:
            pass


# Streamlit UI
st.title("🤖 Pose Detection without OpenCV")

webrtc_streamer(
    key="pose-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=PoseProcessor,
)
"""
