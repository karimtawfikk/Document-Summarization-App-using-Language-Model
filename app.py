import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.document_loaders import PyPDFLoader,  DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import os

#Model & Tokenizer
checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,torch_dtype=torch.float32,low_cpu_mem_usage=False)


#Summarization Pipeline
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        "summarization",
        model=base_model,
        tokenizer=tokenizer,
        max_length=30, #set to small since summarizing chuncks not the entire text
        min_length=15,
        device=0 if torch.cuda.is_available() else -1
    )

    loader = PyPDFLoader(filepath)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50) #each chunk 400 chars, each chunk will overlap the previous one by 50 characters
    chunks = text_splitter.split_documents(pages) #splits into small chuncks in each page 

    all_summaries = []
    for chunk in chunks:
        input_text = chunk.page_content
        result = pipe_sum(input_text)[0]['summary_text']
        all_summaries.append(result)

    final_summary = " ".join(all_summaries)
    return final_summary

#Display PDF Function
@st.cache_data
def get_pdf_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def displayPDF(file_path):
    with open(file_path, "rb") as f:
        st.download_button(
            label="📄 Download Uploaded PDF",
            data=f,
            file_name=os.path.basename(file_path),
            mime="application/pdf"
        )

#Streamlit code
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
                summary = llm_pipeline(filepath)
                st.success(summary)

if __name__ == "__main__":
    main()
