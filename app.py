from streamlit_webrtc import webrtc_streamer
from sign_detector import Recognizer
import av


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

recognizer = Recognizer()

def video_frame_callback(frame):
    global recognizer
    img = frame.to_ndarray(format="bgr24")
    img = recognizer.process_frame(img)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="sample", video_frame_callback=video_frame_callback)