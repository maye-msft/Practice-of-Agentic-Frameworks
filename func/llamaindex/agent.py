from promptflow.core import tool
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
import os
from dotenv import load_dotenv
import chainlit as cl
load_dotenv()

class LlamaIndexMathAgent:
    def __init__(self) -> None:
                
        def multiply(a: int, b: int) -> int:
            """Multiply two integers and returns the result integer"""
            return a * b


        multiply_tool = FunctionTool.from_defaults(fn=multiply)

        def add(a: int, b: int) -> int:
            """Add two integers and returns the result integer"""
            return a + b


        add_tool = FunctionTool.from_defaults(fn=add)


        def minus(a: int, b: int) -> int:
            """Minus two integers and returns the result integer"""
            return a - b


        minus_tool = FunctionTool.from_defaults(fn=minus)


        def divide(a: int, b: int) -> int:
            """Divide one integer by the other and returns the result integer"""
            return a / b


        divide_tool = FunctionTool.from_defaults(fn=divide)


        llm = AzureOpenAI(
            model="gpt-4",
            deployment_name=os.environ["CHAT_MODEL_DEPLOYMENT_NAME"],
            api_key=os.environ["OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["OPENAI_API_VERSION"]
        )
            
        self.agent = ReActAgent.from_tools([multiply_tool, add_tool, minus_tool, divide_tool], llm=llm, verbose=True)

    def execute(self, question: str) -> str:
        response = self.agent.chat(question)
        return response.response
    
    async def aexecute(self, question: str, cl:cl) -> str:
        msg = cl.Message(content="")
        response = self.agent.stream_chat(question)
        msg = cl.Message(content="", author="Assistant")
        for token in response.response_gen:
            await msg.stream_token(token)
        await msg.send()
    
    
