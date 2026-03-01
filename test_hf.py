import os
import logging
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEndpoint

logging.basicConfig(level=logging.INFO)
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    print("NO TOKEN FOUND")
    exit(1)

try:
    print("Testing HuggingFaceEndpoint...")
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", temperature=0.1, huggingfacehub_api_token=hf_token)
    response = llm.invoke("Hi")
    print("SUCCESS: ", response)
except Exception as e:
    print("ERROR DURING INVOKE:", e)
