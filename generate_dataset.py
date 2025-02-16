

import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential

import re


client = ChatCompletionsClient(
    endpoint="https://models.github.ai/inference",
    credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
)


brainrot_word_gen_prompt = "Can you generate brainrot words for me? That is - modern tiktok slang. Include ONLY the words in your response."


brainrot_words_list = []
brainrot_text = ""
brainrot_file = "brainrot.txt"

brainrot_word_response = client.complete(
    messages=[
        UserMessage(brainrot_word_gen_prompt),
    ],
    model="DeepSeek-R1",
    max_tokens=4096,
)
match_list = re.search(r"</think>\s*(.*)", brainrot_word_response.choices[0].message.content, re.DOTALL)
result = match_list.group(1) if match_list else ""

print(result)



brainrot_text_gen_prompt = f"""
I am trying to create a text dataset to train a model on brainrot, that is, gen-z/gen alpha modern TikTok slang. Some examples for these words are: 
        {result}


I want the text generated to be natural , phrases or chunks of cohesive text that contain these words.

 Since I need this dataset to be big, I will use this prompt multiple times and aggregate the results - so be creative with your response. ONLY include the text in your response to speed up cleaning and processing times. Do not include any quotation marks or emojis in your response - only the text. Thanks!

"""

for i in range(5):
    brainrot_text_response = client.complete(
        messages=[
            UserMessage(brainrot_text_gen_prompt),
        ],
        model="DeepSeek-R1",
        max_tokens=4096,
    )
    match_text = re.search(r"</think>\s*(.*)", brainrot_text_response.choices[0].message.content, re.DOTALL)
    new_brainrot_text = "\n" + match_text.group(1) if match_text else ""


    print(new_brainrot_text)

    with open(brainrot_file, "a") as fd: 
        fd.write(new_brainrot_text)



