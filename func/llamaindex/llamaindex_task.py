from promptflow.core import tool
from agent import LlamaIndexMathAgent

agent = LlamaIndexMathAgent()

@tool
def llama_task(question : str) -> str:
    res = agent.execute(question)
    return res
    
