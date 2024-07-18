import os
from dotenv import load_dotenv
from openai import OpenAI
from .helper.system_prompt import SYSTEM_PROMPT

# Load environment variables from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


# summarise the top k embeddings ranked by the reranker and add them to single string for the llm prompt
def getTextForLlm(df, k=7):
    text = ""
    for i in df[0:k]["TextChunk"]:
        text += i
        text += "\n\n"
    return text

# Generate the final response to the user with an LLM 
def getLlmRespone(prompt, json_mode=False):

    if json_mode:
        response_format = "json_object"
        systemPrompt = SYSTEM_PROMPT + """designed to output JSON."""
    else:
        response_format = "text"
        systemPrompt = SYSTEM_PROMPT

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={"type":  response_format},
        messages=[
            {"role": "system", "content": systemPrompt},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content



