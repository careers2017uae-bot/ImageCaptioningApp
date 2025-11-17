import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import requests

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(
    page_title="Image Captioning App",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Image Captioning App by Engr. Bilal")
st.markdown("""
Upload an image and get a descriptive caption.  
You can optionally refine the caption using `llama-3.3-70b-versatile` via Groq API.
""")

# ---------------------------
# BLIP-2 Model Initialization
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip_model()

# ---------------------------
# GROQ API Key
# ---------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY not set. LLM enhancement will be disabled.")

# ---------------------------
# File Upload
# ---------------------------
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

enhance_with_llm = st.checkbox("Enhance caption with llama-3.3-70b-versatile", value=False)

# ---------------------------
# BLIP-2 Caption Function
# ---------------------------
def generate_blip_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# ---------------------------
# LLM Enhancement Function
# ---------------------------
def enhance_caption_llm(caption_text):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are an expert AI assistant. Refine and enhance the following image caption to be more detailed, vivid, and descriptive.  
Keep the original meaning but enrich the objects, scene, colors, mood, and context.

Caption:
\"\"\"{caption_text}\"\"\"
"""

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            result = response.json()
            return result['choices'][0]['message']['content']
        except:
            return "⚠️ Error parsing Groq response."
    else:
        return f"⚠️ Groq API Error: {response.status_code} - {response.text}"

# ---------------------------
# Main Processing
# ---------------------------
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating BLIP-2 caption..."):
            blip_caption = generate_blip_caption(image)
        st.subheader("BLIP-2 Caption")
        st.write(f"📝 {blip_caption}")

        # Optional LLM Enhancement
        if enhance_with_llm:
            if not GROQ_API_KEY:
                st.warning("⚠️ Cannot enhance caption: GROQ_API_KEY not set.")
            else:
                with st.spinner("Enhancing caption with llama-3.3-70b-versatile..."):
                    enhanced_caption = enhance_caption_llm(blip_caption)
                st.subheader("Enhanced Caption")
                st.write(f"✨ {enhanced_caption}")
