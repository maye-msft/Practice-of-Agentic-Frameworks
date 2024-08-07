from promptflow.core import tool

from langchain.chat_models import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import LLMChain

from promptflow.connections import AzureOpenAIConnection
from dotenv import load_dotenv

import os

if "OPENAI_API_KEY" not in os.environ:
    # load environment variables from .env file
    load_dotenv()

llm = AzureChatOpenAI(
    deployment_name=os.environ["CHAT_MODEL_DEPLOYMENT_NAME"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],        
    openai_api_type=os.environ["OPENAI_API_TYPE"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    temperature=0,
)

# Create a MessagesPlaceholder for the chat history
history_placeholder = MessagesPlaceholder("history")

# Construct the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are smart agent to answer any question."),
    history_placeholder,
    ("user", "{input}")
])
    
@tool
def langhcian_task(question : str, chat_history : list) -> str:
    chain = LLMChain(llm=llm, prompt=prompt_template, output_key="metrics")
    res = chain({"input": question, "history": format_chat_history(chat_history)})
    return res["metrics"]
    
def format_chat_history(chat_history):
    formatted_chat_history = []
    for message in chat_history:
        if "inputs" in message:
            formatted_chat_history.append(("user", message["inputs"]["question"]))
        if "outputs" in message:
            formatted_chat_history.append(("system", message["outputs"]["answer"]))
    return formatted_chat_history
