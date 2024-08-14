import os
import argparse
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main(pdf: str) -> str:
    
    load_dotenv()

    loader = PyPDFLoader(pdf)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)


    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_EMBEDDING_MODEL_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_EMBEDDING_MODEL_OPENAI_API_VERSION"],
    )

    Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./index")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a PDF file for embeddings.")
    parser.add_argument("pdf", type=str, help="Path to the PDF file to process")
    args = parser.parse_args()

    main(args.pdf)