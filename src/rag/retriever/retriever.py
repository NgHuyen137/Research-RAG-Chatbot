from typing import List
from ...db import collection
from pymilvus import AnnSearchRequest, RRFRanker


class CustomMultiQueryRetriever:
  def __init__(self, queries: List[str], config: dict):
    self.queries = queries
    self.config = config

  def rerank_documents(self, query: str, docs: List[str], top_k: int=5) -> List[str]:
    """
      Calculate a relevance score between each query-document pair & Return top-k relevant documents 
    """
    bge_rf = self.config["configurable"]["rerank_function"]

    results = bge_rf(
      query=query,
      documents=docs,
      top_k=top_k
    )

    top_k_docs = [result.text for result in results]
    return top_k_docs
   

  def retrieve_documents(self, query: str) -> List[str]:
    documents = []

    # Embed queries into vectors
    bge_m3_ef = self.config["configurable"]["embedding_function"]
    query_embeddings = bge_m3_ef([query])

    # Set up params for dense retrieval
    dense_search_param = {
      "data": query_embeddings["dense"],
      "anns_field": "dense",
      "param": {
        "metric_type": "COSINE"
      },
      "limit": 10
    }
    request_1 = AnnSearchRequest(**dense_search_param)

    # Set up params for sparse retrieval
    sparse_search_param = {
      "data": query_embeddings["sparse"],
      "anns_field": "sparse",
      "param": {
        "metric_type": "IP",
        "params": {"drop_ratio_build": 0.2}
      },
      "limit": 10
    }
    request_2 = AnnSearchRequest(**sparse_search_param)
    reqs = [request_1, request_2]

    # Perform Hybrid search
    results = collection.hybrid_search(
      reqs=reqs,
      rerank=RRFRanker(60),
      limit=10,
      output_fields=["text"]
    )
    
    for result in results[0]:
      documents.append(result["entity"]["text"])

    # Rerank using BGE reranker 
    top_k_docs = self.rerank_documents(query, documents)
    return top_k_docs
  
  
  def get_unique_documents(self, docs: List[str]) -> List[str]:
    return [doc for i, doc in enumerate(docs) if doc not in docs[:i]]
  

  def get_relevant_documents(self) -> List[str]:
    documents = [] # List of top-k documents of each query
    for query in self.queries:
      documents.extend(self.retrieve_documents(query))
    return self.get_unique_documents(documents)
  