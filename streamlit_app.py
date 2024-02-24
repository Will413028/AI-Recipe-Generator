import streamlit as st
from PIL import Image
from openai import OpenAI
import cv2
from utils import load_model, process_image
from config import OPENAI_API_KEY

st.set_page_config(page_title="蔬食智能食譜", page_icon="📸")
client = OpenAI(api_key=OPENAI_API_KEY)


@st.cache_resource(ttl=10800)
def get_model():
    return load_model()


def generate_recipe(ingredients, cuisine):
    with st.spinner('正在生成食譜中，請稍候...'):
        try:
            prompt = f"Based on these ingredients: {ingredients}, generate a {cuisine} recipe. Followed by a list of the ingredients, The ingredients list should be comprehensive and include all necessary items for the recipe."

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Please write in Traditional Chinese language."}
                ])
            return ''.join([choice.message.content for choice in response.choices])
        except Exception as e:
            st.error(f"生成食譜時發生錯誤: {e}")
            return "無法生成食譜，請稍後再試。"


def display_image_and_detect_ingredients(uploaded_file, model):
    if uploaded_file is not None:
        with st.spinner('圖片處理中...'):
            image = Image.open(uploaded_file)
            st.image(image, width=300)
            processed_image, ingredients_detected = process_image(image, model)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            return ingredients_detected
    return []


def main():
    st.title("蔬食智能食譜")
    with st.spinner('模型載入中，請稍候...'):
        model = get_model()

    uploaded_file = st.file_uploader("請上傳一張照片或使用手機拍照", type=["jpg", "jpeg", "png"])
    cuisine = st.selectbox("選擇料理種類", ["台灣料理", "日本料理"])

    if uploaded_file:
        ingredients_detected = display_image_and_detect_ingredients(uploaded_file, model)
        if ingredients_detected:
            recipe = generate_recipe(ingredients_detected, cuisine)
            st.write(recipe)
        else:
            st.write("沒有辨識到任何食物，請嘗試其他照片")
    else:
        st.markdown("<p style='font-size: 16px;'>上傳照片後，您的食譜將在此生成</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()