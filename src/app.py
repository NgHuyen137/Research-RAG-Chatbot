import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

import torch
from uuid import uuid4
import streamlit as st
from src.utils import dump_json
from src.workflow import get_graph_builder
from src.rag import (
  embed_pdf, 
  get_llm, 
  get_embedding_function, 
  get_rerank_function
)
from langchain.schema import HumanMessage

torch.classes.__path__ = []

def create_config() -> dict:
  llm = get_llm()
  embedding_function = get_embedding_function()
  rerank_function = get_rerank_function()
  config = {
    "configurable": {
      "thread_id": str(uuid4()),
      "llm": llm,
      "embedding_function": embedding_function,
      "rerank_function": rerank_function
    }
  }
  return config


def main():
  # Create new session
  if "config" not in st.session_state:
    config = create_config()
    st.session_state.config = config

    graph_builder = get_graph_builder()
    st.session_state.graph_builder = graph_builder


  st.title("AI Researcher")
  st.caption("Your helpful AI assistant, ready to answer questions related to the given research paper.")

  # Upload PDF file
  uploaded_file = st.file_uploader(label="Upload PDF file",type="pdf")

  if "messages" in st.session_state:
    # Load previous messages
    for msg in st.session_state.messages:
      st.chat_message(msg["role"]).write(msg["content"])
  else:
    # Initialize messages
    st.session_state["messages"] = []


  # Show a processing indicator while embedding the PDF
  if uploaded_file and "is_embedded" not in st.session_state:
    with st.spinner("Processing..."):
      embed_pdf(uploaded_file, st.session_state.config)
      st.session_state.is_embedded = True 

    
  # Check if there's a prompt
  if prompt := st.chat_input():
    # Display the user's input on the screen
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if uploaded_file is None:
      st.info("Please upload your research paper to continue! 😉")
      st.stop()
    
    input_message = HumanMessage(content=prompt)
    response_placeholder = st.chat_message("assistant").empty()

    response = ""
    for msg, metadata in st.session_state.graph_builder.stream({"messages": [input_message]}, st.session_state.config, stream_mode="messages"):
      if metadata["langgraph_node"] == "chatbot":
        response = msg.content
        response_placeholder.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Store results
    data = dict()
    data["query"] = prompt
    data["response"] = response  
    dump_json(data)
    

if __name__ == "__main__":
  main()
