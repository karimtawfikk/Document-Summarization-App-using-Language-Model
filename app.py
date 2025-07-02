import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64

# Model & Tokenizer
checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, low_cpu_mem_usage=True)

# Move model from meta to physical device
if next(base_model.parameters()).device.type == "meta":
    base_model.to_empty(device=device)
else:
    base_model.to(device)

st.write("Model is on device:", next(base_model.parameters()).device)

# Summarization Pipeline
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        "summarization",
        model=base_model,
        tokenizer=tokenizer,
        max_length=30,
        min_length=15,
        device=0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
    )

    loader = PyPDFLoader(filepath)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)
    all_summaries = []
    for chunk in chunks:
        input_text = chunk.page_content
        result = pipe_sum(input_text)[0]['summary_text']
        all_summaries.append(result)
    final_summary = " ".join(all_summaries)
    return final_summary

# Display PDF Function
@st.cache_data
def get_pdf_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def displayPDF(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit code
st.set_page_config(layout="wide", page_title="Summarization App")

def main():
    os.makedirs("data", exist_ok=True)
    st.title("Document Summarization App using Language Model")
    uploaded_file = st.file_uploader("Upload your PDF File", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            filepath = "data/" + uploaded_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded PDF File")
                displayPDF(filepath)
            with col2:
                st.info("Summarization is below")
                try:
                    summary = llm_pipeline(filepath)
                    st.success(summary)
                except Exception as e:
                    st.error(f"Error during summarization: {str(e)}")

if __name__ == "__main__":
    main()
