import os
import chainlit as cl
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.chat_engine import SimpleChatEngine

llm = AzureOpenAI(
    model="gpt-4",
    deployment_name=os.environ["CHAT_MODEL_DEPLOYMENT_NAME"],
    api_key=os.environ["OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"]
)

@cl.on_chat_start
async def start():
    chat_engine = SimpleChatEngine.from_defaults(llm=llm)
    chat_engine = cl.user_session.set("chat_engine", chat_engine)


@cl.on_message
async def main(message: cl.Message):
    chat_engine = cl.user_session.get("chat_engine") 
    response = chat_engine.stream_chat(message.content)
    msg = cl.Message(content="", author="Assistant")
    for token in response.response_gen:
         await msg.stream_token(token)
    await msg.send()

