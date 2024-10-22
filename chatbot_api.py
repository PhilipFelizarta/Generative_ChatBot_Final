import tensorflow as tf
import transformers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import create_optimizer

def get_fine_tuned():
	model = TFGPT2LMHeadModel.from_pretrained("./fine_tuned_model_tf")
	tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model_tf")
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = 'left'  # Set left-padding for decoder-only models

	return model, tokenizer

def get_standard():
	model = TFGPT2LMHeadModel.from_pretrained('gpt2')
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = 'left'  # Set left-padding for decoder-only models

	return model, tokenizer

def model_infer(prompt, model, tokenizer, MAX_TOKENS=128):
	# Tokenize the input and create attention mask
	inputs = tokenizer(prompt, padding="max_length", 
					   truncation=True, 
					   max_length=1000, 
					   return_tensors="tf")
	
	input_ids = inputs["input_ids"]
	attention_mask = inputs["attention_mask"]
	
	# Set pad_token_id explicitly to eos_token
	generated_output = model.generate(
		input_ids,
		attention_mask=attention_mask,  # Pass attention mask to handle padding properly
		max_new_tokens=MAX_TOKENS,  # Increase length to allow more fluid continuation
		num_return_sequences=1,  # Generate number of sequences
		no_repeat_ngram_size=1,  # No repetition for natural flow
		do_sample=True,  # Enable sampling
		top_k=40,  # Allows selection only from top_k,
		top_p=0.85,  # Tightens the probability distribution for less randomness
		temperature=0.5,  # Lower temperature for more deterministic responses
		pad_token_id=tokenizer.eos_token_id  # Set pad_token to eos_token explicitly
	)
	
	# Decode the generated output
	generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
	new_generated_text = generated_text[len(prompt):].strip()
	
	return new_generated_text


class ChatBot():
	def __init__(self, fine_tuned=False, max_tokens=20):
		if fine_tuned:
			self.model, self.tokenizer = get_fine_tuned()
		else:
			self.model, self.tokenizer = get_standard()

		# The base prompt is a few shot prompt
		self.base_context = [
			"System: You are a romantic male character responding to a female love interest. You should respond with affection, warmth, and passion.",
			"Female: Why won't you just leave?",
			"Male: I've never felt anything like this before. I know you feel the same way.",
			"Female: How could you?",
			"Male: I thought I could live without you. I was wrong. I need you more than ever.",
			"Female: Are you mad at me?",
			"Male: I'm only frustrated that you can't see how much I love you."
		]

		self.context = self.base_context.copy()
		self.max_tokens = max_tokens

	def reset_context(self):
		self.context = self.base_context.copy()

	def chat(self, prompt):
		engineered_prompt = "Female: " + prompt + "\nMale: "
		self.context.append(engineered_prompt)

		# Join the prompts into a single string
		super_prompt = "\n".join(self.context)
		print("Super Prompt: ", super_prompt)
		model_output = model_infer(super_prompt, self.model, self.tokenizer, MAX_TOKENS=self.max_tokens)

		# The model's prior response should be placed in the context
		new_context = engineered_prompt + model_output
		self.context[-1] = new_context

		print("New Context: ", new_context)

		return model_output

