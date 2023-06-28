from streamlit_webrtc import webrtc_streamer
import av
import time
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector

import streamlit as st

st.set_page_config(
    page_title="American sign language recognition app",
    page_icon="✌️",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.title("American sign language recognition app! ✌️")

hands_detector = HandDetector()

def video_frame_callback(frame):
    global hands_detector
    img = frame.to_ndarray(format="bgr24")

    hands, img = hands_detector.findHands(img)
    # img2 = pose_detector.findPose(img)
    # lmList, bboxInfo = pose_detector.findPosition(img, bboxWithHands=False)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="sample", video_frame_callback=video_frame_callback)