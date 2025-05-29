import re
import fitz
import pymupdf4llm
from typing import Union, List
from ...db import collection
from ...utils import clean_text
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pdf(
  pdf_file: Union[str, bytes],
  chunk_size: int = 1000,
  chunk_overlap: int = 100, 
  min_chunk_size: int = 100
) -> List[str]:
  """
    Extract text from a PDF file & split it into smaller chunks
  """
  # Open the PDF file
  doc=None
  if (isinstance(pdf_file, str) and "data/" in pdf_file):
    doc = fitz.open(filename=pdf_file, filetype="pdf")
  else:
    doc = fitz.open(stream=pdf_file, filetype="pdf")

  # Extract text from the PDF file
  extracted_text = pymupdf4llm.to_markdown(doc=doc)
  # Remove References and Acknowledgments parts
  extracted_text = re.sub(r"#+\s+\*\*\s*(References|Acknowledgments).*?(?=#+\s+\*\*|$)", "", extracted_text, flags=re.IGNORECASE | re.DOTALL)
  # Remove \n between digits
  extracted_text = re.sub(r"(\d+)([\.\,])\s*\n+\s*(\d+)", r"\1\2\3", extracted_text)

  # Split the text into smaller chunks
  splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, 
    chunk_overlap=chunk_overlap,
    is_separator_regex=True,
    add_start_index=True,
    separators=[
      r"\n*#{2,6}\s+(?:\*{2})?(?:[A-Z]|\d+).+(?:\*{2})?", # Split by headings (H2, H3, H4, etc.)
      r"\n*(?<![ \t\r\f\v\w#,*])\s*(?:\*{2})?\d+(?!\s*[kK])(?:\.\d+)*(?:\*{2})?\s+(?:\*{2})?[A-Z].+(?:\*{2})?", # Split by number headings 
      r"\n*-----", # Split by pages
      r"\n*Table\s*\d+:\s*[A-Z].+", # Split by tables
      r"\n{3,}",
      "\n\n",
      "\n",
      " "
    ]
  )
  chunks = splitter.create_documents([extracted_text])
  chunks = [clean_text(chunk.page_content) for chunk in chunks if len(chunk.page_content) > min_chunk_size]
  return chunks


def embed_pdf(
  pdf_file: Union[str, bytes], 
  config: dict, 
  chunk_size: int = 1000,
  chunk_overlap: int = 100, 
  min_chunk_size: int = 100
):
  """
    Embed the PDF file for information retrieval
  """
  # Chunk the PDF file
  chunks = chunk_pdf(pdf_file, chunk_size, chunk_overlap, min_chunk_size)

  # Embed text chunks into vectors
  bge_m3_ef = config["configurable"]["embedding_function"]
  embeddings = bge_m3_ef(chunks)

  # Add to vector database
  entities = [
    chunks,
    embeddings["sparse"],
    embeddings["dense"]
  ]

  collection.insert(entities)
  collection.flush()

  