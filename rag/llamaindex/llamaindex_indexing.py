import os
from typing import List
from promptflow.core import tool
from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from dotenv import load_dotenv

load_dotenv()


text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)

from llama_index.core import Settings

Settings.text_splitter = text_splitter

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=os.environ["EMBEDDING_MODEL_DEPLOYMENT_NAME"],
    api_key=os.environ["OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"],
)

@tool
def index_documents(is_index: bool) -> str:
    if is_index:
        reader = SimpleDirectoryReader("./data")
        documents = reader.load_data()

        index = VectorStoreIndex.from_documents(
            documents, embed_model=embed_model, transformations=[text_splitter], show_progress=True
        )

        index.storage_context.persist(persist_dir="./index")
    
    return "done"
