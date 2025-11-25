import streamlit as st
import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from peft import PeftModel

# --- CONFIG ---
BASE_MODEL_NAME = "unsloth/Llama-3.2-1B-bnb-4bit"  # base model from UnsLoth
HF_LORA_CHECKPOINT = "Jeppcode/ScalableLab2"        # Hugging Face repo
HF_SUBFOLDER = "checkpoint100"                      # LoRA checkpoint subfolder

MAX_SEQ_LENGTH = 2048
DTYPE = "float16"
LOAD_IN_4BIT = True  # reduce memory usage

st.set_page_config(page_title="LoRA Fine-Tuned Demo", layout="wide")

# --- Hugging Face token from Streamlit secrets ---
HF_TOKEN = st.secrets["HF_TOKEN"]

# --- HEADER ---
st.title("🔥 Creative LoRA Demo")
st.markdown("""
This app demonstrates inference with your **fine-tuned LoRA model** on top of the base model.
Compare the **base model** vs your **LoRA-adapted model** for creative tasks like Q&A, summarization, or text generation.
""")

# --- DEVICE ---
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Running on **{device.upper()}**")

# --- LOAD MODELS ---
@st.cache_resource(show_spinner=True)
def load_models():
    st.info("Loading base model...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        token=HF_TOKEN
    )
    base_model.to(device)

    st.info("Applying LoRA adapter from Hugging Face...")
    tuned_model = PeftModel.from_pretrained(
        base_model,
        HF_LORA_CHECKPOINT,
        subfolder=HF_SUBFOLDER,
        use_auth_token=HF_TOKEN
    )
    tuned_model.to(device)

    # Enable faster inference in Unsloth
    FastLanguageModel.for_inference(tuned_model)

    return base_model, tuned_model, tokenizer

base_model, tuned_model, tokenizer = load_models()

# --- USER INPUT ---
st.sidebar.header("Prompt Options")
task = st.sidebar.selectbox("Choose a task", ["Creative Writing", "Q&A", "Summarize Text"])
prompt_input = st.text_area("Enter your prompt here:", "Write a short story about a friendly robot in Paris.")

temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0)
max_tokens = st.sidebar.slider("Max Tokens", 64, 512, 128)
top_p = st.sidebar.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.9)

# --- GENERATE FUNCTION ---
def generate_response(model, prompt: str):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )
    return tokenizer.batch_decode(outputs)[0]

# --- RUN INFERENCE ---
if st.button("Generate"):
    st.info("Generating text from the fine-tuned LoRA model...")
    response = generate_response(tuned_model, prompt_input)
    st.success("Done!")
    st.subheader("Output from LoRA-adapted model:")
    st.write(response)

    st.info("Generating text from the base model for comparison...")
    base_response = generate_response(base_model, prompt_input)
    st.subheader("Output from base model:")
    st.write(base_response)
