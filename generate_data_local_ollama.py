import requests 
import json 
import re
import os

####################
## Defining Constants and Variables
OLLAMA_URL = "http://localhost:11434/api/generate"
brainrot_words = ""
brainrot_text = ""
brainrot_file = "brainrot.txt"
####################


####################
## Word generation 
def generate_brainrot_words():
    brainrot_word_gen_prompt = "Can you generate brainrot words for me? That is - modern tiktok slang. Include ONLY the words in your response - DO NOT FORMAT; JUST HAVE A LIST OF WORDS SEPARATED BY COMMAS."

    payload_word_gen = {
        "model": "deepseek-r1:8b",
        "prompt": brainrot_word_gen_prompt,
        "stream": False
    }

    word_gen_response = requests.post(OLLAMA_URL, json=payload_word_gen)

    word_gen_data = word_gen_response.json()

    match_list = re.search(r"</think>\s*(.*)", word_gen_data.get("response"), re.DOTALL)
    curr_words = match_list.group(1) if match_list else ""

    print(curr_words)
    return curr_words

####################


####################
## Text Generation 
def generate_brainrot_text():
    brainrot_words = generate_brainrot_words()
    brainrot_text_gen_prompt = f"""
        I am trying to create a text dataset to train a model on brainrot, that is, gen-z/gen alpha modern TikTok slang. Some examples for these words are: 
                {brainrot_words}


        I want the text generated to be natural , phrases or chunks of cohesive text that contain these words.
        DO NOT INCLUDE ANY QUOTATION MARKS OR EMOJIS - ONLY THE TEXT. DO NOT NUMBER OR PROVIDE AN INTRO.

        Since I need this dataset to be big, I will use this prompt multiple times and aggregate the results - so be creative with your response. ONLY include the text in your response to speed up cleaning and processing times. ONCE AGAIN - DO NOT INCLUDE any quotation marks or emojis in your response - ONLY THE TEXT. Thanks!
    """

    payload_text_gen = {
        "model": "deepseek-r1:8b",
        "prompt": brainrot_text_gen_prompt,
        "stream": False
    }

    text_gen_response = requests.post(OLLAMA_URL, json=payload_text_gen)

    text_gen_data = text_gen_response.json()

    match_list = re.search(r"</think>\s*(.*)", text_gen_data.get("response"), re.DOTALL)
    curr_text = match_list.group(1) if match_list else ""

    print(curr_text)
    return curr_text

####################


####################
## Write to file

generate_brainrot_text()


####################
