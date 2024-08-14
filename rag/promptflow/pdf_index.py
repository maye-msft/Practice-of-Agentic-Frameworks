import PyPDF2
import faiss
import os

from pathlib import Path
from embedding import OAIEmbedding
from faiss_indexer import FAISSIndexer

chunk_size = int(os.environ.get("CHUNK_SIZE"))
chunk_overlap = int(os.environ.get("CHUNK_OVERLAP"))
    
def create_faiss_index(pdf_path: str) -> FAISSIndexer:
    file_name = f"pdf.index_{chunk_size}_{chunk_overlap}"
    index_persistent_path = Path("./.index/") / file_name
    index_persistent_path = index_persistent_path.resolve().as_posix()
    index = FAISSIndexer()
    if not os.path.exists(index_persistent_path):
        os.makedirs(index_persistent_path)
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        segments = split_text(text, chunk_size, chunk_overlap)
        index.insert_batch(segments)
        index.save(index_persistent_path)
    else:
        index.load(index_persistent_path)
    return index


# Split the text into chunks with CHUNK_SIZE and CHUNK_OVERLAP as character count
def split_text(text, chunk_size, chunk_overlap):
    # Calculate the number of chunks
    num_chunks = (len(text) - chunk_overlap) // (chunk_size - chunk_overlap)

    # Split the text into chunks
    chunks = []
    for i in range(num_chunks):
        start = i * (chunk_size - chunk_overlap)
        end = start + chunk_size
        chunks.append(text[start:end])

    # Add the last chunk
    chunks.append(text[num_chunks * (chunk_size - chunk_overlap):])

    return chunks

def query_text(index: FAISSIndexer, text: str, top_k=5) -> list:
    return index.query(text, top_k)

if __name__ == "__main__":
    pdf_path = "../../data/2023_canadian_budget.pdf"
    index = create_faiss_index(pdf_path)
    results = search_text(index, "What is the total amount of the 2023 Canadian federal?")
    for result in results:
        print(result.text, result.score)
        print("----")