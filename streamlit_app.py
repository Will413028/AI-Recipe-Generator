import cv2
import torch
from super_gradients.training import models
import numpy as np
import math
import streamlit as st
from PIL import Image


api_key = st.secrets["OPENAI_API_KEY"]

classNames = ['avocado', 'beans', 'beet', 'bell pepper', 'broccoli', 'brus capusta', 'cabbage', 'carrot', 'cayliflower', 'celery', 'corn', 'cucumber', 'eggplant', 'fasol', 'garlic', 'hot pepper', 'onion', 'peas', 'potato', 'pumpkin', 'rediska', 'redka', 'salad', 'squash-patisson', 'tomato', 'vegetable marrow']

def process_image_and_generate_recipe(image_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.get('yolo_nas_s', num_classes=26, checkpoint_path='./vegetable_yolo_nas.pth').to(device)

    np_image = np.array(image_data)
    frame = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    class_final_names = []

    result = model.predict(frame, conf=0.30)

    if result.prediction is not None:
        for bbox_xyxy, confidence, cls in zip(result.prediction.bboxes_xyxy, result.prediction.confidence, result.prediction.labels):
            bbox = np.array(bbox_xyxy).astype(int)
            class_name = classNames[int(cls)]
            class_final_names.append(class_name)
            conf = math.ceil((confidence*100))/100
            label = f'{class_name}{conf}'
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    return frame


st.set_page_config(page_title="蔬食智能食譜", page_icon="📸")
st.title("蔬食智能食譜")
st.title(api_key)
uploaded_file = st.file_uploader("請上傳一張照片或使用手機拍照", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    
    st.image(image, caption="上傳的照片", use_column_width=True)

    processed_image = process_image_and_generate_recipe(image)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    st.image(processed_image, caption="預測結果", use_column_width=True)
