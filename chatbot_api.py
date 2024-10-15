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

def model_infer(prompt, model, tokenizer, MAX_TOKENS=50):
	# Tokenize the input and create attention mask
	inputs = tokenizer(prompt, padding="max_length", 
					   truncation=True, 
					   max_length=MAX_TOKENS, 
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
		top_k=50,  # Allows selection only from top_k,
		top_p=0.92,  # Tightens the probability distribution for less randomness
		temperature=0.7,  # Lower temperature for more deterministic responses
		pad_token_id=tokenizer.eos_token_id  # Set pad_token to eos_token explicitly
	)
	
	# Decode the generated output
	generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
	new_generated_text = generated_text[len(prompt):].strip()
	
	return new_generated_text


class ChatBot():
	def __init__(self, fine_tuned=False, max_tokens=50):
		if fine_tuned:
			self.model, self.tokenizer = get_fine_tuned()
		else:
			self.model, self.tokenizer = get_standard()

		self.base_prompt = "Continue the following prompt as a female character from a romance movie. "
		self.context = [self.base_prompt]
		self.max_tokens = max_tokens

	def reset_context(self):
		self.context = [self.base_prompt]

	def chat(self, prompt):
		self.context.append(prompt)

		# Join the prompts into a single string
		super_prompt = " ".join(self.context)
		new_context = model_infer(self.context, self.model, self.tokenizer, MAX_TOKENS=self.max_tokens)

		# The model's prior response should be placed in the context
		self.context.append(new_context)

		return new_context

