import chainlit as cl
import os

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()
from llamaindex_indexing import INDEX_PATH
from llama_index.core import Settings
from llama_index.core.service_context import ServiceContext
from llama_index.core.callbacks import CallbackManager

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)

try:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)
    # load index
    index = load_index_from_storage(storage_context)
except:
    documents = SimpleDirectoryReader(INDEX_PATH).load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist()
    
@cl.on_chat_start
async def on_chat_start():
    Settings.llm = AzureOpenAI(
        model="gpt-4",
        deployment_name=os.environ["CHAT_MODEL_DEPLOYMENT_NAME"],
        api_key=os.environ["OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["OPENAI_API_VERSION"],
        streaming=True
    )
    

    Settings.embed_model = AzureOpenAIEmbedding(
        model=os.environ["EMBEDDING_MODEL_DEPLOYMENT_NAME"],
        deployment_name=os.environ["EMBEDDING_MODEL_DEPLOYMENT_NAME"],
        api_key=os.environ["OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["OPENAI_API_VERSION"],
    )
    
    Settings.context_window = 4096

    service_context = ServiceContext.from_defaults(callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]))
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=2, service_context=service_context)
    cl.user_session.set("query_engine", query_engine)


@cl.on_message
async def on_message(message: cl.Message):
    query_engine = cl.user_session.get("query_engine") # type: RetrieverQueryEngine

    msg = cl.Message(content="", author="Assistant")
    
    res = await cl.make_async(query_engine.query)(message.content)

    for token in res.response_gen:
        print (token)
        print("************")
        if token != "":
            await msg.stream_token(token)
    await msg.send()
    
    


