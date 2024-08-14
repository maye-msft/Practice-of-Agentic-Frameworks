import chainlit as cl
from agent import LlamaIndexMathAgent

@cl.on_chat_start
async def on_chat_start():
    agent = LlamaIndexMathAgent()
    cl.user_session.set("agent", agent)


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: LlamaIndexMathAgent
    res = agent.execute(message.content)
    await cl.Message(content=res).send()
    
    


