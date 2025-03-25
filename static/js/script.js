document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("send-btn").addEventListener("click", sendMessage);
    document.getElementById("user-input").addEventListener("keypress", function (event) {
        if (event.key === "Enter") sendMessage();
    });
    document.getElementById("theme-toggle").addEventListener("click", toggleMode);
    document.getElementById("voice-btn").addEventListener("click", startVoiceRecognition);

    // Load chat history on page load
    loadChatHistory();
});

function sendMessage() {
    var userInput = $("#userInput").val().trim();
    if (userInput === "") return; // Prevent empty messages

    // Append user message to chatbox
    appendMessage("You", userInput);

    // Save user message to database
    saveMessage("You", userInput);

    // Clear input field
    $("#userInput").val("");

    // Simulate bot response (Replace this with actual API call)
    setTimeout(() => {
        var botResponse = `Here is your response for "${userInput}"`;
        appendMessage("Bot", botResponse);
        saveMessage("Bot", botResponse); // Save bot response
    }, 1000);
}

function appendMessage(sender, message) {
    $("#chatbox").append(`<p><strong>${sender}:</strong> ${message}</p>`);
    scrollChatToBottom();
}

function scrollChatToBottom() {
    var chatbox = $("#chatbox");
    chatbox.scrollTop(chatbox.prop("scrollHeight"));
}

// Save message to backend
function saveMessage(sender, message) {
    $.post("/save_message", { sender: sender, message: message });
}

// Load chat history
function loadChatHistory() {
    $.get("/get_chat_history", function (data) {
        data.forEach((msg) => {
            appendMessage(msg.sender, msg.message);
        });
    });
}

function toggleMode() {
    document.body.classList.toggle("dark-mode");
    let button = document.getElementById("modeToggle");
    button.innerHTML = document.body.classList.contains("dark-mode") ? "☀️ Light Mode" : "🌙 Dark Mode";
}
