import os
from promptflow.core import tool
from llama_index.core import StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from dotenv import load_dotenv
from llamaindex_indexing import INDEX_PATH

load_dotenv()

embed_model = AzureOpenAIEmbedding(
    model=os.environ["EMBEDDING_MODEL_DEPLOYMENT_NAME"],
    deployment_name=os.environ["EMBEDDING_MODEL_DEPLOYMENT_NAME"],
    api_key=os.environ["OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"],
)

from llama_index.core import Settings

llm = AzureOpenAI(
    model="gpt-4",
    deployment_name=os.environ["CHAT_MODEL_DEPLOYMENT_NAME"],
    api_key=os.environ["OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"],
)
Settings.llm = llm



@tool
def query(question: str, is_index: str) -> str:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)

    # load index
    index = load_index_from_storage(storage_context)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
        embed_model=embed_model,
    )

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer()

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
    )
    response = query_engine.query(question)

    return response.response