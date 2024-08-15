import os
import argparse
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ.get("EMBEDDING_MODEL_DEPLOYMENT_NAME"),
    openai_api_version=os.environ.get("OPENAI_API_VERSION")
)

INDEX_PATH = "../.langchain-index"

def main(pdf: str) -> str:
    create_faiss_index(pdf)
    
def create_faiss_index(pdf_path: str) -> FAISS:
    
    load_dotenv()

    if not os.path.exists(INDEX_PATH):
        os.makedirs(INDEX_PATH)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        chunk_size = int(os.environ.get("CHUNK_SIZE"))
        chunk_overlap = int(os.environ.get("CHUNK_OVERLAP"))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(docs)

        # Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./index")
        faiss = FAISS.from_documents(documents=splits, embedding=embeddings)
        faiss.save_local(INDEX_PATH)
        return faiss
    else:
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a PDF file for embeddings.")
    parser.add_argument("pdf", type=str, help="Path to the PDF file to process")
    args = parser.parse_args()

    main(args.pdf)