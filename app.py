import streamlit as st
import os
import requests
import base64

# Page setup
st.set_page_config(
    page_title="🖼️ Pro Image Captioning",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Professional Image Captioning App")
st.markdown("""
Upload an image, and Groq API will describe it in detail with multiple captions.  
It also provides context about objects, scene, colors, and mood.  
""")

# Load API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found. Please set it as an environment variable.")
    st.stop()

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Function to describe image using Groq
def describe_image_groq(image_bytes, num_variations=3):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    prompt = f"""
You are an expert AI assistant. Given the image in base64 below, provide {num_variations} detailed, descriptive captions.  
Include objects, scene, actions, colors, and mood if applicable. Make each caption unique.  

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

# Display image preview and analyze button
if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    if st.button("Generate Descriptions"):
        with st.spinner("Generating captions with Groq..."):
            image_bytes = uploaded_image.read()
            descriptions = describe_image_groq(image_bytes)
        st.subheader("Generated Captions")
        st.markdown(descriptions)
