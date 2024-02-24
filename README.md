# AI-Recipe-Generator

The AI Recipe Generator is an application built on Streamlit and OpenAI, designed to automatically generate recipes from uploaded images of ingredients. This app leverages advanced image processing technologies and AI models to identify ingredients in images and generate recipes accordingly.

## Running the Application

Before running the application, ensure the environment

- Python 3.9

### 1. Install dependencies using the following command

```bash
pip install -r requirements.txt
```

### 2. Setting Up Streamlit Secrets

#### Create a Secrets File

```bash
AI-Recipe-Generator/
├── .streamlit/
│   └── secrets.toml
├── streamlit_app.py
└── ...
```

#### Add your OPENAI_API_KEY inside the secrets.toml file

```bash
OPENAI_API_KEY = "your_openai_api_key_here"
```

### 3. Running the Application

```bash
streamlit run streamlit_app.py
```

## How to Use

1. Once the app is running, your browser will automatically open to the app's main page.
2. Upload an image containing ingredients.
3. Select the type of cuisine you want to generate a recipe for.
4. After uploading the image and selecting a cuisine type, the system will automatically identify the ingredients in the image and generate a corresponding recipe.
