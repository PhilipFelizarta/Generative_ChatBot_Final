from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the models and tokenizers globally
fine_tuned_model, fine_tuned_tokenizer = None, None
standard_model, standard_tokenizer = None, None
fine_tuned_chatbot = None
standard_chatbot = None

def get_fine_tuned():
    print("Loading fine-tuned model...")
    model = TFGPT2LMHeadModel.from_pretrained("./fine_tuned_model_tf")
    tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model_tf")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    print("Fine-tuned model loaded successfully.")
    return model, tokenizer

def get_standard():
    print("Loading standard GPT-2 model...")
    model = TFGPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    print("Standard model loaded successfully.")
    return model, tokenizer

def model_infer(prompt, model, tokenizer, MAX_TOKENS=50):
    try:
        inputs = tokenizer(prompt, padding="max_length", truncation=True, max_length=MAX_TOKENS, return_tensors="tf")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        generated_output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_TOKENS,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
        print("Generated output:", generated_output)  # Debug print

        generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
        new_generated_text = generated_text[len(prompt):].strip()
        return new_generated_text
    except Exception as e:
        print("Error during inference:", e)
        return "An error occurred during text generation."

class ChatBot:
    def __init__(self, model, tokenizer, max_tokens=50):
        self.model = model
        self.tokenizer = tokenizer
        self.base_prompt = "Continue the following prompt as a female character from a romance movie."
        self.context = [self.base_prompt]
        self.max_tokens = max_tokens

    def reset_context(self):
        self.context = [self.base_prompt]
        print("Context has been reset.")

    def chat(self, prompt):
        try:
            self.context.append(prompt)
            super_prompt = " ".join(self.context)
            new_context = model_infer(super_prompt, self.model, self.tokenizer, MAX_TOKENS=self.max_tokens)
            self.context.append(new_context)
            return new_context
        except Exception as e:
            print("Error during chat:", e)
            return "An error occurred while generating a response."

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
        fine_tuned_model, fine_tuned_tokenizer = get_fine_tuned()
        standard_model, standard_tokenizer = get_standard()
        fine_tuned_chatbot = ChatBot(fine_tuned_model, fine_tuned_tokenizer)
        standard_chatbot = ChatBot(standard_model, standard_tokenizer)
        print("Models loaded and chatbots initialized.")
    except Exception as e:
        print("Error loading models:", e)
    
    app.run(host="0.0.0.0", port=5222)
