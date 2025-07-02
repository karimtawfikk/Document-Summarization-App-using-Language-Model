import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64

#Model & Tokenizer
checkpoint = "LaMini-Flan-T5-248M"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)


#Summarization Pipeline
def llm_pipeline(filepath):
    loader = PyPDFLoader(filepath)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)  #each chunk 400 chars, each chunk will overlap the previous one by 50 characters
    chunks = text_splitter.split_documents(pages) #splits into small chuncks in each page

    inputs = [chunk.page_content for chunk in chunks]
    inputs_tokenized = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    summaries = base_model.generate(
        **inputs_tokenized,
        max_length=35,
        min_length=15,
        num_beams=4
    )

    decoded = tokenizer.batch_decode(summaries, skip_special_tokens=True)
    return " ".join(decoded)


#Display PDF Function
@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

#Streamlit code
st.set_page_config(layout="wide", page_title="Summarization App")

def main():
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
