import streamlit as st
import pymongo

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["chatbot_db"]
collection = db["chat_history"]

# Streamlit UI
st.set_page_config(page_title="Knowledge Buddy 🤖", layout="wide")
st.title("Knowledge Buddy 🤖")
st.write("Hello! How may I assist you today?")

# Store chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "username" not in st.session_state:
    st.session_state.username = st.text_input("Enter your name to start chatting:", key="username_input")
    if st.session_state.username:
        st.success(f"Welcome, {st.session_state.username}! 🎉")
else:
    st.title(f"Welcome {st.session_state.username}! 👋")

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_input = st.text_input("Type your message here...", key="user_input")

if user_input:
    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Store user message in MongoDB
    collection.insert_one({"role": "user", "message": user_input})

    # Simulate bot response (Replace this with actual chatbot logic)
    bot_response = f"Processing... Here is your response for '{user_input}'"
    
    # Append bot response to chat history
    st.session_state.messages.append({"role": "bot", "content": bot_response})
    
    # Store bot response in MongoDB
    collection.insert_one({"role": "bot", "message": bot_response})

    # Display bot response
    with st.chat_message("bot"):
        st.write(bot_response)
