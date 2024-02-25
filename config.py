import streamlit as st


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
CLASS_NAMES = ['avocado', 'beans', 'beet', 'bell pepper', 'broccoli', 'brus capusta', 'cabbage', 'carrot', 'cayliflower', 'celery', 'corn', 'cucumber', 'eggplant', 'fasol', 'garlic', 'hot pepper', 'onion', 'peas', 'potato', 'pumpkin', 'rediska', 'redka', 'salad', 'squash-patisson', 'tomato', 'vegetable marrow']
CUISINE_OPTIONS = {
    "台灣料理": "Taiwanese",
    "日本料理": "Japanese",
    "意大利料理": "Italian",
    "法國料理": "French",
    "韓國料理": "Korean",
    "西班牙料理": "Spanish"
    }
MODEL_CHECKPOINT_PATH = './vegetable_yolo_nas.pth'