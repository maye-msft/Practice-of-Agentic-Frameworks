from promptflow.core import tool
from pathlib import Path
import faiss
import os
from dotenv import load_dotenv
load_dotenv()

from pdf_index import create_faiss_index, query_text

pdf_path = "../../data/2023_canadian_budget.pdf"
index = create_faiss_index(pdf_path)

@tool
def find_context_tool(question: str)->list:
    result = query_text(index=index, text=question, top_k=5)
    return [c.text for c in result]