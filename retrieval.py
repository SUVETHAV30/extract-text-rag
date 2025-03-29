import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from text_utils import extract_text_from_pdf, extract_text_from_docx, load_csv_data
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


PDF_PATH = "data/sample.pdf"
DOCX_PATH = "data/sample.docx"
CSV_PATH = "data/Students_Grading_Dataset.csv"


pdf_text = extract_text_from_pdf(PDF_PATH)
docx_text = extract_text_from_docx(DOCX_PATH)
csv_data = load_csv_data(CSV_PATH)


csv_text = "\n".join([f"{col}: {csv_data[col].astype(str).unique()[:5]}" for col in csv_data.columns])


all_text_data = pdf_text + "\n" + docx_text + "\n" + csv_text


def generate_embeddings(text_list):
    """Generate embeddings using the transformer model."""
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :].numpy()
    return embeddings


def create_faiss_index():
    """Creates and stores FAISS index from extracted text."""
    text_chunks = all_text_data.split("\n")
    embeddings = generate_embeddings(text_chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, text_chunks


def search_faiss(query, index, text_chunks):
    """Retrieves most relevant text chunks for a given query."""
    query_embedding = generate_embeddings([query])
    _, indices = index.search(query_embedding, k=3)  
    return [text_chunks[i] for i in indices[0]]
