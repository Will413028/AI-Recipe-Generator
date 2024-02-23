import streamlit as st
from PIL import Image

st.set_page_config(page_title="手機拍照示例", page_icon="📸")

st.title("手機拍照並顯示照片")

uploaded_file = st.file_uploader("請上傳一張照片或使用手機拍照", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    
    st.image(image, caption="上傳的照片", use_column_width=True)
