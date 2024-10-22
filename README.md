# Romantic Generative AI Chatbot Using Advanced Language Processing Techniques

## Abstract
Modern advancements in the field of generative AI have enabled an unfathomable range of text and language based applications. Open access to pre-trained transformer-based large language models in particular, has empowered small teams with modest computing power to create highly tailored applications. This project exploits the Hugging Face transformers library (GPT-2) and the publicly available Cornell Movie Dialogs Corpus database to create a ChatBot that mimics the jargon and dialogue of a romantic male movie character.

## Running the Code
The Chatbot can be interacted with by running `python chatbot_interface.py` Once your computer is running this program, open the `chatbot.html` file. The HTML file will allow you to interact with our Romantic Male Chatbot via a text interface that includes a text bar, model-selection, and a button to reset the context of the conversation.

While running `chatbot_interface.py` you will see print statements showing what prompts are actually being fed to the fine-tuned or standard models. This is denoted as the "super-prompt". You'll also see the new context you are adding when conversating with the chatbot.

## Library Structure

### API
`chatbot_interface.py` utilizes the chatbot_api.py file create instances of a class object that tracks and engineers context for the chatbot. You'll see that the class handles importing the fine tuned or standard models and has few-shot prompt engineering techniques within it.

`chatbot_interface.py` and `chatbot.html` were created by Fuad. `chatbot_api.py` was created by Philip

### Model Training
The fine tuning and dataset construction for GPT-2 Takes place in the Model Training Pipeline.ipynb. Here you will see how we use pandas to isolate only romantic movies with male characters from the Cornell Corpus. You'll notice that this jupyter notebook produces a file that is later used for EDA.

`Model Training Pipeline.ipynb` was created by Philip

### EDA and Preprocessing
All Exploratory Data Analysis will be found in the EDA.ipynb file.

`EDA.ipynb` was created by Gabriel.

The file to isolate English only movies is the `AAI-520-pre-processing.ipynb` and was created by Fuad.
