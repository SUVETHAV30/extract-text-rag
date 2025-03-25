import fitz  # PyMuPDF for PDF processing
import docx
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file"""
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text() for page in doc])
    return text

def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file"""
    doc = docx.Document(docx_path)
    text = " ".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_csv(csv_path):
    """Extracts text from a CSV file"""
    df = pd.read_csv(csv_path)
    return " ".join(df.astype(str).values.flatten())

def get_embeddings(text):
    """Generates embeddings from text"""
    return model.encode(text)

