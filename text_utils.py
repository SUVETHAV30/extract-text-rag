import pandas as pd
import PyPDF2
import docx

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF files."""
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def extract_text_from_docx(docx_path):
    """Extract text from DOCX files."""
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_csv_data(csv_path):
    """Load CSV as Pandas DataFrame."""
    return pd.read_csv(csv_path)

def query_csv_data(df, query):
    """Dynamically query CSV data based on user input."""
    query = query.lower()

    if "average" in query:
        for col in df.select_dtypes(include=['number']).columns:
            if col.lower() in query:
                return f"The average {col} is {df[col].mean():.2f}"

    if "highest" in query or "maximum" in query:
        for col in df.select_dtypes(include=['number']).columns:
            if col.lower() in query:
                return f"The highest {col} is {df[col].max()}"

    if "lowest" in query or "minimum" in query:
        for col in df.select_dtypes(include=['number']).columns:
            if col.lower() in query:
                return f"The lowest {col} is {df[col].min()}"

    if "top" in query:
        for col in df.columns:
            if "gpa" in col.lower():
                top_students = df.nlargest(3, col)[["Student_Name", col]]
                return f"Top students based on {col}:\n{top_students.to_string(index=False)}"

    return "I couldn't find relevant information in the dataset."




