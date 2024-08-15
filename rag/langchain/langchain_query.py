import os
from promptflow.core import tool
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_indexing import create_faiss_index

load_dotenv()

model = AzureChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["CHAT_MODEL_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
)


pdf_path = "../../data/2023_canadian_budget.pdf"
vectorstore = create_faiss_index(pdf_path)

@tool
def query(question: str) -> str:

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


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain.invoke(question)
