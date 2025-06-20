# Research RAG Chatbot
This project aims to develop a Vietnamese RAG chatbot that could answer questions related to a given research paper.

## Technologies used

- **LLM Frameworks:** Langchain, Langgraph.
- **Large Language Models:** Google Gemini.
- **Vector Database:** Milvus. 

## Performance

![Metric Comparison](image/result.png)

## Streamlit UI

![Streamlit UI](image/streamlit_ui.png)

## Setup

1. **Install Milvus Standalone**
- Follow this link to install: [Link](https://milvus.io/docs/install_standalone-windows.md)
- Run Milvus.

2. **Clone the repository**
```
git clone <repository_url>
```

3. **Create virtual environment**
```
python -m venv venv
venv/Scripts/activate
```

4. **Install required libraries**
```
pip install requirements.txt
```

3. **Run Streamlit app**
```
cd src
streamlit run app.py
```
