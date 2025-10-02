import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF warnings

import cv2
import mediapipe as mp
import tempfile
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


# ========== Guard Check Function ==========
def check_guard(img, wrist, shoulder, elbow, h, w):
    wrist_xy = (int(wrist.x * w), int(wrist.y * h))
    elbow_xy = (int(elbow.x * w), int(elbow.y * h))
    shoulder_xy = (int(shoulder.x * w), int(shoulder.y * h))

    if wrist.y < shoulder.y:
        color = (0, 255, 0)  # green
    else:
        color = (0, 0, 255)  # red

    cv2.line(img, shoulder_xy, elbow_xy, color, 3)
    cv2.line(img, elbow_xy, wrist_xy, color, 3)

    cv2.circle(img, shoulder_xy, 6, color, -1)
    cv2.circle(img, elbow_xy, 6, color, -1)
    cv2.circle(img, wrist_xy, 6, color, -1)


# ========== Pose Processor for Live ==========
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark

            # left arm
            check_guard(img,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST],
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                        h, w)

            # right arm
            check_guard(img,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST],
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
                        h, w)
        else:
            cv2.putText(img, "No pose detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame.from_ndarray(img, format="bgr24")

    def __del__(self):
        try:
            self.pose.close()
        except:
            pass


# ========== Streamlit UI ==========
st.title("Pose Detection (Live or Uploaded Video)")

option = st.radio("Choose video source:", ["Live Webcam", "Upload Video"])

if option == "Live Webcam":
    webrtc_streamer(
        key="pose-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=PoseProcessor,
    )

elif option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        # Save uploaded file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        input_path = tfile.name

        # Output file
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        pose = mp_pose.Pose()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                landmarks = results.pose_landmarks.landmark

                # left arm
                check_guard(frame,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST],
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                            h, w)

                # right arm
                check_guard(frame,
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST],
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
                            h, w)
            else:
                cv2.putText(frame, "No pose detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            out.write(frame)

        cap.release()
        out.release()
        pose.close()

        # Reload processed video in Streamlit
        st.video(out_path)
