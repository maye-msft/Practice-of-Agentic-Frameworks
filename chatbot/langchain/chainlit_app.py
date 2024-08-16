from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl

from dotenv import load_dotenv
load_dotenv()
import os

@cl.on_chat_start
async def on_chat_start():
    model = AzureChatOpenAI(
        deployment_name=os.environ["CHAT_MODEL_DEPLOYMENT_NAME"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],        
        openai_api_type=os.environ["OPENAI_API_TYPE"],
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        temperature=0,
        streaming=True
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are smart agent to answer any question.
                chat_history
                {chat_history}"""
            ),
            ("user", "{question}"),
            
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("history",[])
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")
    history = cl.user_session.get("history") 
    
    history_str = ""
    for h in history:
        history_str += f"{h[0]}: {h[1]}\n"
        
    async for chunk in runnable.astream(
        {"question": message.content, "chat_history": history_str},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        
        await msg.stream_token(chunk)


    await msg.send()    
    history.append(("user", message.content))
    history.append(("system", msg.content))
