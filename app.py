import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import base64

checkpoint = "facebook/bart-large-cnn"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)

def llm_pipeline(filepath):
    loader = PyPDFLoader(filepath)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)

    summaries = []
    for chunk in chunks:
        input_text = "Summarize the following text: " + chunk.page_content.strip().replace("\n", " ")
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)

        output = model.generate(
            **inputs,
            max_length=180, 
            min_length=60,
            num_beams=4,
            early_stopping=True
        )

        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)

    
    return " ".join(summaries)
    
@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

st.set_page_config(layout="wide", page_title="Summarization App")

def main():
    st.title("Document Summarization App using BART-Large-CNN")

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
                st.info("Summarized Content")
                summary = llm_pipeline(filepath)
                st.success(summary)

if __name__ == "__main__":
    main()
