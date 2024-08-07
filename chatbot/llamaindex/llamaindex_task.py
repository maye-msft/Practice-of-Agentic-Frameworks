from promptflow.core import tool
from promptflow.connections import AzureOpenAIConnection
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.chat_engine import SimpleChatEngine

from dotenv import load_dotenv
load_dotenv()

import os

llm = AzureOpenAI(
    model="gpt-4o",
    deployment_name=os.environ["CHAT_MODEL_DEPLOYMENT_NAME"],
    api_key=os.environ["OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"],
)
    
chat_engine = SimpleChatEngine.from_defaults(llm=llm)

@tool
def llama_task(question : str) -> str:
    response = chat_engine.chat(
        question
    )
    return response.response
    
