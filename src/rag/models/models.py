import os
from dotenv import load_dotenv
from pymilvus.model.reranker import BGERerankFunction
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

def get_llm():
  llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model="gemini-2.0-flash",
    max_retries=10
  )
  return llm

def get_embedding_function():
  return BGEM3EmbeddingFunction(model_name="BAAI/bge-m3", device="cpu", use_fp16=False)

def get_rerank_function():
  return BGERerankFunction(model_name="BAAI/bge-reranker-v2-m3", device="cpu")
