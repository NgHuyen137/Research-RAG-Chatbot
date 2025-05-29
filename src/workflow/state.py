from typing import List
from langgraph.graph import MessagesState

# Define the state of the graph
class State(MessagesState):
  rewritten_queries: str = [] # The rewritten query of the original query
  retrieved_docs: List[str] = [] # The most relevant documents to the query
  summary: str = "" # A summary of the conversation
