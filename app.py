import streamlit as st
from PIL import Image
from io import BytesIO
import requests
import os

st.set_page_config(page_title="Image Captioning App", page_icon="🖼️", layout="centered")
st.title("🖼️ Image Captioning App (Hugging Face API + Optional LLaMA)")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
enhance_with_llm = st.checkbox("Enhance caption with LLaMA (via Groq)", value=False)

# Load API keys from secrets
HF_API_KEY = st.secrets.get("HF_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

# Hugging Face API endpoint for BLIP-2
HF_API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"


def generate_caption_hf(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(HF_API_URL, headers=headers, files={"file": img_bytes})

    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        return f"⚠️ Hugging Face API Error: {response.status_code} {response.text}"

def enhance_caption_llm(caption_text):
    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY not set."
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"Refine and enhance the following image caption to be more vivid and detailed:\n\n{caption_text}"
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            return response.json()['choices'][0]['message']['content']
        except:
            return "⚠️ Error parsing Groq response."
    else:
        return f"⚠️ Groq API Error: {response.status_code} - {response.text}"

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            caption = generate_caption_hf(image)
        st.subheader("Caption from Hugging Face")
        st.write(f"📝 {caption}")

        if enhance_with_llm:
            with st.spinner("Enhancing caption with LLaMA..."):
                enhanced_caption = enhance_caption_llm(caption)
            st.subheader("Enhanced Caption")
            st.write(f"✨ {enhanced_caption}")
