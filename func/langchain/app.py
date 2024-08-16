import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig
from agent import LangChainMathAgent

@cl.on_chat_start
async def on_chat_start():
    agent = LangChainMathAgent()
    cl.user_session.set("prompt_history",[])
    cl.user_session.set("agent", agent)


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: LangChainMathAgent
    prompt_history = cl.user_session.get("prompt_history")  # type: list
    res = await agent.aexecute(message.content, prompt_history, cl)
    prompt_history.append(("user", message.content))
    prompt_history.append(("system", res))

    
    


