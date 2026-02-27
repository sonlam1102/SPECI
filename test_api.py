import json
import openai

LST_KEY = [
    {
        "API_KEY": "bd475283dc23429e89e9ac97445fb912",
        "AZURE_ENDPOINT": "https://vietgpt.openai.azure.com/",
        "API_VERSION": "2023-07-01-preview",
        "GPT_MODEL": "gpt-4o-2024-05-13"
    },
    {
        "API_KEY": "9ddfc1c22ed24caea354e56e7c5d1474",
        "AZURE_ENDPOINT": "https://swede-central.openai.azure.com/",
        "API_VERSION": "2024-09-01-preview",
        "GPT_MODEL": "gpt-4o-mini"
    },
    {
        "API_KEY": "15a078f95d524d53aa9ba48223e22926",
        "AZURE_ENDPOINT": "https://eastus2instancefranck.openai.azure.com/",
        "API_VERSION": "2024-12-01-preview",
        "GPT_MODEL": "o3-mini-2025-01-31"
    },
]

global KEY_OPEN_API
KEY_OPEN_API = LST_KEY[2]

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
    api_key=KEY_OPEN_API["API_KEY"],
    azure_endpoint=KEY_OPEN_API["AZURE_ENDPOINT"],
    api_version=KEY_OPEN_API["API_VERSION"]
)



def query(system_prompt, user_prompt, temp=0.3, max_token=512):
    messages = [
        {"role": "user", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=KEY_OPEN_API["GPT_MODEL"],
        messages=messages,
        max_completion_tokens=max_token,
        temperature=temp,
    )
    print(response)
    text = response.choices[0].message.content
    return text


if __name__ == '__main__':
    print(KEY_OPEN_API)
    print(query("Generate a warm greeting. ", "Hello", max_token=40, temp=1))