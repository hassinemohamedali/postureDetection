import cv2
import mediapipe as mp
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Mediapipe pose & drawing
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def check_guard(img, wrist, shoulder, elbow, h, w):
    # Convert normalized landmarks to pixel coordinates
    wrist_xy = (int(wrist.x * w), int(wrist.y * h))
    elbow_xy = (int(elbow.x * w), int(elbow.y * h))
    shoulder_xy = (int(shoulder.x * w), int(shoulder.y * h))

    # Check if wrist is above shoulder
    if wrist.y < shoulder.y:
        color = (0, 255, 0)  # green
    else:
        color = (0, 0, 255)  # red

    # Draw custom lines (shoulder → elbow → wrist)
    cv2.line(img, shoulder_xy, elbow_xy, color, 3)
    cv2.line(img, elbow_xy, wrist_xy, color, 3)

    # Optional: draw circles
    cv2.circle(img, shoulder_xy, 6, color, -1)
    cv2.circle(img, elbow_xy, 6, color, -1)
    cv2.circle(img, wrist_xy, 6, color, -1)

class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

            check_guard(img, left_wrist, left_shoulder, left_elbow, h, w)
            check_guard(img, right_wrist, right_shoulder, right_elbow, h, w)
        else:
            cv2.putText(img, "No pose detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def __del__(self):
        try:
            self.pose.close()
        except:
            pass

st.title("🤖 Pose Detection with Streamlit + WebRTC")

webrtc_streamer(
    key="pose-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=PoseProcessor,
)
