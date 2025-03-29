import torch
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from flask import Flask, render_template, request, jsonify
import PyPDF2
import docx
import pandas as pd
import re
import faiss
from sentence_transformers import SentenceTransformer, util
import pytesseract  
import cv2  
from pymongo import MongoClient
import datetime
import os

app = Flask(__name__)

client = MongoClient("mongodb://localhost:27017/")  
db = client["chatbot_memory"]
collection = db["chat_history"]



pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
extracted_texts = ""

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_text_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    extracted_text = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        text = pytesseract.image_to_string(frame)  
        extracted_text += text + "\n"

    cap.release()
    return extracted_text.strip()



@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"response": "No video uploaded!"})

    video_file = request.files["video"]
    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(video_path)
    
    
    extracted_text = extract_text_from_video(video_path)
    return jsonify({"response": extracted_text if extracted_text else "No text found in the video."})


    
    return jsonify({"response": f"Video '{video_file.filename}' uploaded successfully!"})



MODEL_NAME = "deepset/roberta-base-squad2"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)


embedding_model = SentenceTransformer(EMBEDDING_MODEL)

dimension = 384  
index = faiss.IndexFlatL2(dimension)

documents = [] 


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_data_from_csv(csv_path):
    return pd.read_csv(csv_path)


def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    text = pytesseract.image_to_string(gray)
    return text.strip()




pdf_text = extract_text_from_pdf("data/sample.pdf")
docx_text = extract_text_from_docx("data/sample.docx")
csv_data = extract_data_from_csv("data/Students_Grading_Dataset.csv")


academic_context = pdf_text + "\n" + docx_text

# Extract CSV column names
csv_columns = csv_data.columns.tolist()
csv_column_embeddings = embedding_model.encode(csv_columns, convert_to_tensor=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    try:
        user_input = request.json["msg"]
        print(f"User Query: {user_input}")  
        
 

        
        if extracted_texts:
            response = qa_pipeline(question=user_input, context=extracted_texts)
            bot_reply = response["answer"]

        
        best_column = find_best_matching_column(user_input)
        print(f"Best Matched Column: {best_column}")

        if best_column:
            response = process_csv_query(user_input, best_column)
            return jsonify({"response": response})

       
        response = qa_pipeline(question=user_input, context=academic_context)
        bot_reply = response["answer"]

        if not bot_reply or bot_reply.strip() == "":
            return jsonify({"response": "I couldn't find any relevant information."})
        
        return jsonify({"response": bot_reply})

    except Exception as e:
        return jsonify({"response": f"Error generating response: {str(e)}"})
    
@app.route('/ask', methods=['POST'])
def ask():
    """Answer queries based on extracted text."""
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    
    response = process_query(query)  

    return jsonify({"response": response})



@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        file = request.files["file"]
        file_path = f"uploads/{file.filename}"
        file.save(file_path)

        if file.filename.endswith(".pdf"):
            extracted_text = extract_text_from_pdf(file_path)

        elif file.filename.endswith(".docx"):
            extracted_text = extract_text_from_docx(file_path)

        elif file.filename.endswith(".csv"):
            global csv_data, csv_columns, csv_column_embeddings
            csv_data = extract_data_from_csv(file_path)
            csv_columns = csv_data.columns.tolist()
            csv_column_embeddings = embedding_model.encode(csv_columns, convert_to_tensor=True)
            return jsonify({"response": "CSV uploaded and updated successfully."})

        elif file.filename.endswith((".jpg", ".png", ".jpeg")):
            extracted_text = extract_text_from_image(file_path)




        return jsonify({"response": f"Extracted Text: {extracted_text[:500]}"})  # Limit preview

    except Exception as e:
        return jsonify({"response": f"Error processing file: {str(e)}"})
    
def find_best_matching_column(query):
    query = query.lower().strip()

    for col in csv_columns:
        if col.lower() in query:
            return col

    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, csv_column_embeddings)

    best_match_idx = torch.argmax(similarities).item()
    best_match_score = similarities[0][best_match_idx].item()

    if best_match_score > 0.6:
        return csv_columns[best_match_idx]

    return None

def process_csv_query(query, column_name):
    try:
        if column_name not in csv_data.columns:
            return f"Column {column_name} not found."

        if "average" in query:
            return f"The average {column_name} is {csv_data[column_name].mean():.2f}"

        
        elif "highest" in query or "max" in query:
            return f"The highest {column_name} is {csv_data[column_name].max()}"

        elif "lowest" in query or "min" in query:
            return f"The lowest {column_name} is {csv_data[column_name].min()}"

        elif "top" in query:
            top_values = csv_data.nlargest(3, column_name)[[column_name]]
            return f"Top 3 values for {column_name}:\n{top_values.to_string(index=False)}"

        elif "count" in query or "how many" in query:
            unique_count = csv_data[column_name].nunique()
            return f"There are {unique_count} unique values in {column_name}"
        
        return f"Summary of {column_name}:\n{csv_data[column_name].describe()}"

    except Exception as e:
        return f"Error processing column '{column_name}': {e}"
    
def store_text_in_faiss(text):
    """Store extracted text as embeddings in FAISS."""
    global documents
    embedding = model.encode([text])
    index.add(embedding)
    documents.append(text)

def process_query(query):
    """Retrieve relevant text from FAISS."""
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, 1) 
    
    if I[0][0] != -1:  
        return documents[I[0][0]]
    
    return "No relevant information found."
    


@app.route('/upload-image', methods=['POST'])
def upload_image():
    """Handle image upload and extract text."""
    global extracted_texts
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    
    image_path = os.path.join("uploads", file.filename)
    file.save(image_path)

    
    extracted_text = extract_text_from_image(image_path)


    return jsonify({"extracted_text": extracted_text})

def save_message(user, message, response):
    """Save user message and bot response to the database"""
    chat_entry = {
        "user": user,
        "message": message,
        "response": response,
        "timestamp": datetime.datetime.utcnow()
    }
    collection.insert_one(chat_entry)

def get_last_message(user):
    """Retrieve the last message from the user"""
    last_message = collection.find_one({"user": user}, sort=[("timestamp", -1)])
    return last_message["message"] if last_message else "No previous messages found."

@app.route("/get", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("msg")
    user = "User1" 

   
    if user_message.lower() in ["what was my last question?", "recall my last message"]:
        bot_response = get_last_message(user)
    else:
        bot_response = f"Processing your request: {user_message}"  # Replace with AI processing logic
    
    save_message(user, user_message, bot_response)
    
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    
    
    
