from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from .nodes import (
  query_routing,
  query_rewrite,
  query_decompose,
  document_retrieval,
  chatbot,
  summarize_conversation,
  should_continue
)
from .state import State


def get_graph_builder():
  """
    Build and return a compiled state graph for a conversational pipeline
  """
  # Initialize short-term memory 
  memory = MemorySaver()

  # Create graph
  graph = StateGraph(State)
  graph.add_node(query_routing)
  graph.add_node(query_rewrite)
  graph.add_node(query_decompose)
  graph.add_node(document_retrieval)
  graph.add_node(chatbot)
  graph.add_node(should_continue)
  graph.add_node(summarize_conversation)

  graph.add_conditional_edges(
    START,
    query_routing,
    {
      "query_rewrite", "query_rewrite",
      "query_decompose", "query_decompose"
    }
  )

  graph.add_edge("query_rewrite", "document_retrieval")
  graph.add_edge("query_decompose", "document_retrieval")
  graph.add_edge("document_retrieval", "chatbot")

  graph.add_conditional_edges(
    "chatbot", 
    should_continue,
    {
      "summarize_conversation": "summarize_conversation",
      END: END
    }
  )

  graph.add_edge("summarize_conversation", END)
  graph_builder = graph.compile(checkpointer=memory)

  return graph_builder
