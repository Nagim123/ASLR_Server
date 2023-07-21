from streamlit_webrtc import webrtc_streamer
from sign_detector import Recognizer
import av
import threading
import streamlit as st
import time

sentences = ""

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
container = st.empty()

def video_frame_callback(frame):
    global recognizer, sentences
    img = frame.to_ndarray(format="bgr24")
    img = recognizer.process_frame(img)
    sentences = recognizer.get_current_sentence()
    return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(key="sample", video_frame_callback=video_frame_callback)

while ctx.state.playing:
    container.text(f"Translated message:{sentences}")
    time.sleep(2)
# if ctx.:
#     st.text("Play!")
# else:
#     st.text("Nothing(")