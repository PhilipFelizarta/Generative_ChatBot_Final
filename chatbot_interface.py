from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from flask_cors import CORS  # Add this import

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and tokenizer globally
fine_tuned_model, fine_tuned_tokenizer = None, None
standard_model, standard_tokenizer = None, None

def get_fine_tuned():
    model = TFGPT2LMHeadModel.from_pretrained("./fine_tuned_model_tf")
    tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model_tf")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return model, tokenizer

def get_standard():
    model = TFGPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return model, tokenizer

def model_infer(prompt, model, tokenizer, MAX_TOKENS=50):
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
    
    generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    new_generated_text = generated_text[len(prompt):].strip()
    return new_generated_text

class ChatBot:
    def __init__(self, fine_tuned=False, max_tokens=50):
        if fine_tuned:
            self.model, self.tokenizer = get_fine_tuned()
        else:
            self.model, self.tokenizer = get_standard()
        self.base_prompt = "Continue the following prompt as a female character from a romance movie."
        self.context = [self.base_prompt]
        self.max_tokens = max_tokens

    def reset_context(self):
        self.context = [self.base_prompt]

    def chat(self, prompt):
        self.context.append(prompt)
        super_prompt = " ".join(self.context)
        new_context = model_infer(super_prompt, self.model, self.tokenizer, MAX_TOKENS=self.max_tokens)
        self.context.append(new_context)
        return new_context

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("prompt", "")
    fine_tuned = data.get("fine_tuned", False)
    chatbot = ChatBot(fine_tuned=fine_tuned)
    response = chatbot.chat(prompt)
    return jsonify({"response": response})

if __name__ == "__main__":
    # Load models when starting the server to reduce delay on first request
    fine_tuned_model, fine_tuned_tokenizer = get_fine_tuned()
    standard_model, standard_tokenizer = get_standard()
    app.run(host="0.0.0.0", port=5222)
