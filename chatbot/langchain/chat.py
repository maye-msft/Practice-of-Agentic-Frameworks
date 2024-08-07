import os

from langchain.chat_models import AzureChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are smart agent to answer any question."),
        ("user", "{input}"),
    ]
)

chain = LLMChain(llm=llm, prompt=prompt, output_key="metrics")
res = chain({"input": "What is ChatGPT?"})
print(res["metrics"])