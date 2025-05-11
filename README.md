RAG-Based Chatbot

Overview
This Retrieval-Augmented Generation (RAG) chatbot is designed to process and extract information from various document types, including PDFs, DOCX, CSVs, images, and videos. It retrieves relevant content from documents and enhances responses using MongoDB for storage and Streamlit for UI.

Features
Document Processing: Extracts and retrieves data from PDF, DOCX, CSV, images, and videos.
Natural Language Processing (NLP): Enhances query understanding and response accuracy.
Chat History Management: Stores previous conversations using MongoDB.
User Authentication: Secure access and session management.
Streamlit UI: Interactive web interface for user-friendly chatbot interaction.
No LangChain Used: Implements custom retrieval and generation logic.

Technologies Used
Frontend: Streamlit
Backend: Python (FastAPI / Flask)
Database: MongoDB
Text Processing: Natural Language Toolkit (NLTK), spaCy
Document Parsing: PyMuPDF, pdfplumber, python-docx, pandas
Image Processing: OpenCV, Tesseract OCR
Video Processing: OpenCV, FFmpeg
Vector Search: FAISS (if required for semantic retrieval)

Setup Instructions
1. Clone the Repository
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot

3. Create a Virtual Environment
python -m venv myenv
source myenv/bin/activate  
myenv\Scripts\activate
   
4. Install Dependencies
pip install -r requirements.txt

5. Run the Chatbot
streamlit run streamlit_app.py

Deployment on Streamlit Cloud
Push your code to GitHub:
git add .
git commit -m "Initial commit"
git push origin main
Go to Streamlit Community Cloud and deploy your app by linking your GitHub repository.

Future Enhancements
Integrate LLM-based retrieval for improved answer generation.
Add speech-to-text support for voice-based queries.
Improve UI with custom themes and interactive elements







