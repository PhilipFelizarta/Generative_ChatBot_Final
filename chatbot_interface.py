from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from flask_cors import CORS
import chatbot_api

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the models and tokenizers globally
fine_tuned_chatbot = None
standard_chatbot = None

@app.route("/chat", methods=["POST"])
def chat():
	data = request.json
	prompt = data.get("prompt", "")
	fine_tuned = data.get("fine_tuned", True)
	
	# Choose the appropriate chatbot based on the fine_tuned flag
	if fine_tuned:
		print("Using fine-tuned model")
		chatbot = fine_tuned_chatbot
	else:
		print("Using standard model")
		chatbot = standard_chatbot

	try:
		response = chatbot.chat(prompt)
	except Exception as e:
		print("Error during chat request:", e)
		response = "An error occurred while processing your request."

	return jsonify({"response": response})

@app.route("/reset_context", methods=["POST"])
def reset_context():
	data = request.json
	fine_tuned = data.get("fine_tuned", True)
	
	# Choose the appropriate chatbot based on the fine_tuned flag
	if fine_tuned:
		print("Resetting context for fine-tuned model")
		fine_tuned_chatbot.reset_context()
	else:
		print("Resetting context for standard model")
		standard_chatbot.reset_context()

	return jsonify({"message": "Context reset successful."})

if __name__ == "__main__":
	# Load models and initialize global chatbot instances
	print("Starting to load models...")
	try:
		fine_tuned_chatbot = chatbot_api.ChatBot(fine_tuned=True)
		standard_chatbot = chatbot_api.ChatBot(fine_tuned=False)
		print("Models loaded and chatbots initialized.")
	except Exception as e:
		print("Error loading models:", e)
	
	app.run(host="0.0.0.0", port=5222)
