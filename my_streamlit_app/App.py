import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# --- CONFIG ---
REPO_ID = "Jeppcode/ScalableLab2"  # Hugging Face repo
FILENAME = "model-q4_k_m.gguf"       # GGUF quantized model file

# Fixed inference parameters
TEMPERATURE = 1.0
MAX_TOKENS = 128
TOP_P = 0.9

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

# --- Generate function ---
def generate_response(prompt_text: str):
    resp = model(
        prompt_text,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P
    )
    return resp["choices"][0]["text"]

# --- Run inference ---
if st.button("Generate"):
    with st.spinner("Generating text..."):
        output = generate_response(prompt)
    st.success("✅ Done!")
    st.subheader("Output:")
    st.write(output)
