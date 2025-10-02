import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF warnings

import cv2
import mediapipe as mp
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# ========== Pose & Drawing ==========
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


# ========== Custom Function ==========
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

    # Optional: draw small circles on joints
    cv2.circle(img, shoulder_xy, 6, color, -1)
    cv2.circle(img, elbow_xy, 6, color, -1)
    cv2.circle(img, wrist_xy, 6, color, -1)


# ========== Processor Class ==========
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        # Create a Pose instance per processor (thread-safe)
        self.pose = mp_pose.Pose()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            # Draw skeleton
            mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

            # Apply guard checks
            check_guard(img, left_wrist, left_shoulder, left_elbow, h, w)
            check_guard(img, right_wrist, right_shoulder, right_elbow, h, w)
        else:
            cv2.putText(img, "No pose detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Return as av.VideoFrame
        new_frame = frame.from_ndarray(img, format="bgr24")
        return new_frame

    def __del__(self):
        try:
            self.pose.close()
        except:
            pass


# ========== Streamlit UI ==========
st.title("Pose Detection with Streamlit + WebRTC")
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
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        input_path = tfile.name

        #output file
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        pose = mp.solutions.pose.Pose()

        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # get landmarks
                landmarks = results.pose_landmarks.landmark
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

                # check guard
                check_guard(frame, left_wrist, left_shoulder, left_elbow, h, w)
                check_guard(frame, right_wrist, right_shoulder, right_elbow, h, w)
            else:
                cv2.putText(frame, "No pose detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show in Streamlit
            out.write(frame)
        

        cap.release()
        out.release()
        pose.close()

        # Display output video
        st.success("Processing complete!")
        st.video(out_path)