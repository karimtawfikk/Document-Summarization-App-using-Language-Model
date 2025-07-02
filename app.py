import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import logging
import sys
import transformers
import langchain

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create data directory
os.makedirs("data", exist_ok=True)

# Model & Tokenizer
checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, low_cpu_mem_usage=True)
    if next(base_model.parameters()).device.type == "meta":
        logger.info("Model on meta device, moving to physical device")
        base_model.to_empty(device=device)
    else:
        base_model.to(device)
    st.write(f"Model is on device: {next(base_model.parameters()).device}")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    logger.error(f"Model loading error: {str(e)}")
    raise e

# Summarization Pipeline
def llm_pipeline(filepath):
    try:
        pipe_sum = pipeline(
            "summarization",
            model=base_model,
            tokenizer=tokenizer,
            max_length=100,  # Increased for better summaries
            min_length=15,
            device=0 if torch.cuda.is_available() else -1,
            do_sample=False,
            num_beams=4,
        )

        loader = PyPDFLoader(filepath)
        pages = loader.load_and_split()
        st.write(f"Loaded {len(pages)} pages from PDF")
        logger.info(f"Loaded {len(pages)} pages from PDF")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = text_splitter.split_documents(pages)
        st.write(f"Split into {len(chunks)} chunks")
        logger.info(f"Split into {len(chunks)} chunks")

        all_summaries = []
        for i, chunk in enumerate(chunks):
            input_text = chunk.page_content.strip()
            logger.info(f"Chunk {i+1} content: {input_text[:100]}...")
            if not input_text or len(input_text.split()) < 5 or input_text.lower().startswith("the the"):
                logger.warning(f"Chunk {i+1} is empty, too short, or repetitive, skipping")
                continue
            try:
                result = pipe_sum(input_text)[0]["summary_text"]
                logger.info(f"Chunk {i+1} summary: {result}")
                all_summaries.append(result)
            except Exception as e:
                logger.error(f"Error summarizing chunk {i+1}: {str(e)}")
                all_summaries.append(f"[Error in chunk {i+1}]")

        if not all_summaries:
            return "Error: No valid summaries generated. Check PDF content."
        final_summary = " ".join(all_summaries)
        return final_summary
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return f"Error in summarization: {str(e)}"

# Cache only the base64 encoding
@st.cache_data
def get_pdf_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        return base64_pdf
    except Exception as e:
        logger.error(f"Error encoding PDF to base64: {str(e)}")
        return None

# Display PDF Function (no widgets in cached function)
def displayPDF(file_path):
    try:
        base64_pdf = get_pdf_base64(file_path)
        if base64_pdf:
            st.write(f"Base64 string length: {len(base64_pdf)}")
            logger.info(f"Base64 string length: {len(base64_pdf)}")
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        else:
            st.error("Failed to encode PDF to base64")
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")
        logger.error(f"PDF display error: {str(e)}")
    # Move widget outside cached function
    with open(file_path, "rb") as f:
        st.download_button(
            label="📄 Download PDF",
            data=f,
            file_name=os.path.basename(file_path),
            mime="application/pdf",
        )

# Streamlit code
st.set_page_config(layout="wide", page_title="Summarization App")

def main():
    st.title("Document Summarization App using Language Model")

    # Show environment info
    st.write(f"Python: {sys.version.split()[0]}")
    st.write(f"PyTorch: {torch.__version__}")
    st.write(f"Transformers: {transformers.__version__}")
    st.write(f"LangChain: {langchain.__version__}")

    uploaded_file = st.file_uploader("Upload your PDF File", type=["pdf"])
    if uploaded_file is not None:
        filepath = os.path.join("data", uploaded_file.name)
        try:
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file.flush()
            st.write(f"Saved file size: {os.path.getsize(filepath)} bytes")
            logger.info(f"Saved PDF: {filepath}, size: {os.path.getsize(filepath)} bytes")
        except Exception as e:
            st.error(f"Error saving PDF: {str(e)}")
            logger.error(f"PDF save error: {str(e)}")
            return

        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            with col1:
                st.info("Uploaded PDF File")
                displayPDF(filepath)
            with col2:
                st.info("Summarization is below")
                summary = llm_pipeline(filepath)
                if "Error" in summary:
                    st.error(summary)
                else:
                    st.success(summary)

if __name__ == "__main__":
    main()
