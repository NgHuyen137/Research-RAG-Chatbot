from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langgraph.graph import END

import re
from typing import Any, Dict, List
from .state import State
from ..rag import (
  query_routing_prompt,
  multi_query_decompose_prompt,
  multi_query_rewrite_prompt,
  generate_prompt,
  CustomMultiQueryRetriever
)
from ..utils import clean_text


class LineListOutputParser(BaseOutputParser[List[str]]):
  """Output parser for a list of lines."""

  def parse(self, text: str) -> List[str]:
    lines = text.strip().split("\n")
    return list(filter(None, lines))  # Remove empty lines
  

# Define the query routing node
def query_routing(state: State, config: dict):
  llm = config["configurable"]["llm"]
  query = state["messages"][-1].content
  summary = state["summary"] if "summary" in state else ""

  query_routing_chain = query_routing_prompt | llm | JsonOutputParser()
  query_class = query_routing_chain.invoke({"summary": summary, "messages": state["messages"], "query": query})

  if query_class["class"] == "no-retrieve" and len(state["messages"]) >= 3:
    return "chatbot"
  if query_class["class"] == "simple" or (query_class["class"] == "no-retrieve" and len(state["messages"]) < 3):
    return "query_rewrite" 
  if query_class["class"] == "complex":
    return "query_decompose"


# Define the query rewrite node
def query_rewrite(state: State, config: dict) -> Dict[str, Any]:
  llm = config["configurable"]["llm"]
  query = state["messages"][-1].content
  summary = state["summary"] if "summary" in state else ""

  output_parser = LineListOutputParser()
  multi_query_rewrite_chain = multi_query_rewrite_prompt | llm | output_parser

  rewritten_queries = multi_query_rewrite_chain.invoke({"summary": summary, "messages": state["messages"], "query": query})
  rewritten_queries = [clean_text(rewritten_query) for rewritten_query in rewritten_queries]
  return {"rewritten_queries": rewritten_queries}


# Define the query decompose node
def query_decompose(state: State, config: dict) -> Dict[str, Any]:
  llm = config["configurable"]["llm"]
  query = state["messages"][-1].content
  summary = state["summary"] if "summary" in state else ""

  output_parser = LineListOutputParser()
  multi_query_decompose_chain = multi_query_decompose_prompt | llm | output_parser

  rewritten_queries = multi_query_decompose_chain.invoke({"summary": summary, "messages": state["messages"], "query": query})
  rewritten_queries = [clean_text(rewritten_query) for rewritten_query in rewritten_queries]
  return {"rewritten_queries": rewritten_queries}


# Define the document retrieval node
def document_retrieval(state: State, config: dict) -> Dict[str, Any]:
  rewritten_queries = state["rewritten_queries"]
 
  # Retrieve relevant documents
  retriever = CustomMultiQueryRetriever(queries=rewritten_queries, config=config)
  retrieved_docs = retriever.get_relevant_documents()

  return {"retrieved_docs": retrieved_docs}


# Define the chatbot node
def chatbot(state: State, config: dict) -> Dict[str, Any]:
  llm = config["configurable"]["llm"]
  summary = state["summary"] if "summary" in state else ""
  retrieved_docs = state["retrieved_docs"] if "retrieved_docs" in state else ""
  retrieved_docs_text = "Document:\n\n".join([doc for doc in retrieved_docs])

  messages = generate_prompt.format_messages(
    summary=summary,
    retrieved_docs_text=retrieved_docs_text
  )
  messages += state["messages"]
  res = llm.invoke(messages)
  res_text = res.content
  res_text = re.sub(r"```markdown|```", "", res_text).strip()
  return {"messages": [AIMessage(content=res_text)]}


# Define the node to summarize the conversation
def summarize_conversation(state: State, config: dict) -> Dict[str, Any]:
  llm = config["configurable"]["llm"]
  summary = state["summary"] if "summary" in state else ""
  if summary:
    summary_message = (
      f"This is the summary of the conversation to date: {summary}\n"
      "Extend the summary by taking into account the new messages above:"
    )
  else:
    summary_message = "Create a summary of the conversation above:"

  messages = state["messages"][:-2] + [HumanMessage(content=summary_message)]
  response = llm.invoke(messages)
  messages = [RemoveMessage(id=message.id) for message in state["messages"][:-2]] # Retain 2 recent messages
  return {"summary": response.content, "messages": messages}


# Define the node to continue or stop the summarization
def should_continue(state) -> Any:
  maximum_messages = 6 # Summarize the conversation when the number of messages is 6
  messages = state["messages"]
  if len(messages) > maximum_messages:
    return "summarize_conversation"
  return END
