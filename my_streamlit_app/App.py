import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# --- CONFIG ---
REPO_ID = "Jeppcode/ScalableLab2"  # Hugging Face repo
FILENAME = "model-q4_k_m.gguf"       # GGUF quantized model file

st.set_page_config(page_title="CPU GGUF Demo", layout="wide")
st.title("🖥 CPU GGUF Demo")
st.markdown(
    """
This demo runs **CPU-only inference** using a GGUF quantized model directly from Hugging Face.
It demonstrates creative text generation beyond a simple chatbot.
"""
)

# --- Hugging Face token from Streamlit secrets ---
HF_TOKEN = st.secrets["HF_TOKEN"]

# --- Load model (cached) ---
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        use_auth_token=HF_TOKEN
    )
    return Llama(model_path=model_path)

# Show spinner while downloading/loading model
with st.spinner("Downloading and loading GGUF model..."):
    model = load_model()
st.success("✅ Model loaded!")

# --- User input ---
prompt = st.text_area("Enter your prompt here:", "Write a short story about a friendly robot in Paris.")

temperature = st.slider("Temperature", 0.1, 2.0, 1.0)
max_tokens = st.slider("Max tokens", 32, 512, 128)
top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.9)

# --- Generate function ---
def generate_response(prompt_text: str):
    resp = model(
        prompt_text,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )
    # Llama returns a dict with 'choices' containing 'text'
    return resp["choices"][0]["text"]

# --- Run inference ---
if st.button("Generate"):
    with st.spinner("Generating text..."):
        output = generate_response(prompt)
    st.success("✅ Done!")
    st.subheader("Output:")
    st.write(output)
