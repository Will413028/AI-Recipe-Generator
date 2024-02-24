import streamlit as st
from PIL import Image
from openai import OpenAI
import cv2
from utils import load_model, process_image
from config import OPENAI_API_KEY


client = OpenAI(api_key=OPENAI_API_KEY)


@st.cache_resource(ttl=10800)
def get_model():
    return load_model()


def generate_recipe(ingredients, cuisine):
    try:
        prompt = f"Take these ingredients: {ingredients}. Generate a {cuisine} recipe based on these ingredients."
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


def main():
    st.set_page_config(page_title="蔬食智能食譜", page_icon="📸")
    st.title("蔬食智能食譜")
    
    uploaded_file = st.file_uploader("請上傳一張照片或使用手機拍照", type=["jpg", "jpeg", "png"])
    cuisine = st.selectbox("選擇食譜種類", ["台灣料理", "日本料理"])
    st.title("您的食譜將在此生成")
    
    model = get_model()
    
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="上傳食物照片", use_column_width=True)
        processed_image, ingredients_detected = process_image(image, model)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        if ingredients_detected:
            recipe = generate_recipe(ingredients_detected, cuisine)
            st.write(recipe)
        else:
            st.write("沒有辨識到任何食物，請嘗試其他照片")

if __name__ == "__main__":
    main()