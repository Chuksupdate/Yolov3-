# object_detection_app.py

import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load YOLOv8 model (downloaded automatically if not present)
model = YOLO("yolov8n.pt")  # you can also use yolov8s.pt, yolov8m.pt, etc.

st.title("📷 Real-time Object Detection with YOLOv8")
st.markdown("Using **Streamlit + Webcam + YOLOv8**")

# Define the transformer for real-time processing
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLOv8 detection
        results = self.model.predict(img, conf=0.5)

        # Draw boxes on frame
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Streamlit-webrtc component
webrtc_streamer(
    key="object-detection",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
