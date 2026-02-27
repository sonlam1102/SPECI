import openai
import json
import os
from dotenv import load_dotenv
import nltk
import stanza
from misc import *
from google import genai

load_dotenv()
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
nltk.download('wordnet')  # This will download the WordNet data

def parse_chatgpt_json(response_text):
    try:
        # Attempt to parse the entire response as JSON
        response_text = response_text.replace("```json", "")
        response_text = response_text.replace("```", "")
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # If direct parsing fails, attempt to extract JSON from markdown or text
        try:
            start_index = response_text.index('{')
            end_index = response_text.rindex('}') + 1
            json_string = response_text[start_index:end_index]
            data = json.loads(json_string)
        except (ValueError, json.JSONDecodeError) as e:
             return f"Error: Could not parse JSON: {e}"
        
    return data

client = openai.AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("API_VERSION")
)

client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) 

def query(system_prompt, user_prompt, temp=0.3, max_token=512):
    messages = [
        {"role": "user", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=os.getenv("GPT_MODEL"),
        messages=messages,
        max_completion_tokens=max_token,
        temperature=temp,
    )
    text = response.choices[0].message.content
    return text


def query2(user_prompt, temp=0.3, max_token=512):
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=os.getenv("GPT_MODEL"),
        messages=messages,
        max_completion_tokens=max_token,
        temperature=temp,
    )
    text = response.choices[0].message.content
    return text


def query3(user_prompt):
    response = client_gemini.models.generate_content(
        model="gemini-2.5-flash", 
        contents=user_prompt
    )
    return response.text
