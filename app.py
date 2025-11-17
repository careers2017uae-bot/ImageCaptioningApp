import streamlit as st
import os
import requests

# Page config
st.set_page_config(page_title="Image Captioning App", page_icon="🖼️", layout="centered")
st.title("🖼️ Image Captioning App")
st.write("Upload an image, and Groq API will describe it in detail.")

# Load GROQ API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found. Please set it as an environment variable.")
    st.stop()

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Function to call Groq API for image description
def describe_image_groq(image_bytes):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Base64 encode the image for API input
    import base64
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    prompt = f"""
You are an AI assistant. Describe the content of the image provided in base64 below in a detailed, descriptive, and informative way. 
Include objects, actions, scene, colors, and mood if applicable. Do not hallucinate.  

Image (base64):
{image_b64}
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

# Analyze the uploaded image
if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    if st.button("Describe Image"):
        with st.spinner("Generating description with Groq..."):
            image_bytes = uploaded_image.read()
            description = describe_image_groq(image_bytes)
        st.subheader("Image Description")
        st.write(description)
