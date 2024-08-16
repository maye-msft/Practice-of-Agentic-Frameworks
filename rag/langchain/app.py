from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_indexing import create_faiss_index
import chainlit as cl

from dotenv import load_dotenv
load_dotenv()
import os

pdf_path = "../../data/2023_canadian_budget.pdf"
vectorstore = create_faiss_index(pdf_path)

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

    
    
    retriever = vectorstore.as_retriever()

    # Define the template string
    template_string = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # Format the prompt with the context and question
    prompt = ChatPromptTemplate(
        input_variables=['context', 'question'], 
        messages=[
            HumanMessagePromptTemplate(prompt=PromptTemplate(
                input_variables=['context', 'question'], 
                template=template_string))])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    cl.user_session.set("history",[])
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")
    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()    

