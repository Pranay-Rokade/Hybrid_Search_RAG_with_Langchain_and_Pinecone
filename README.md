
# 🔍 Hybrid Search RAG with LangChain and Pinecone

This project demonstrates how to build a **Hybrid Search-based Retrieval-Augmented Generation (RAG)** system using:

- 🧠 **LangChain** for orchestration
- 🔗 **Pinecone** as the vector database (dense + sparse)
- 🧬 **HuggingFace Embeddings** for dense vector representation
- 📊 **BM25 Encoder** from `pinecone-text` for sparse encoding
- 📄 Sample sentence corpus to test hybrid search ranking

---

## ❓ What is Hybrid Search?

**Hybrid Search** combines **dense vector search** (semantic similarity) and **sparse vector search** (keyword/lexical matching like BM25) to improve retrieval accuracy, especially on short queries or factual content.

It leverages:
- Dense Embeddings: to capture *semantic meaning*
- Sparse BM25 (TF-IDF): to capture *exact keyword matches*

---

## 🔧 What is Pinecone?

[Pinecone](https://www.pinecone.io/) is a managed vector database built for real-time, scalable vector search. It supports both **dense** and **sparse** vectors (hybrid search), making it ideal for production-grade RAG pipelines.

---

## 🚀 Features

- 🔍 Dense + Sparse Hybrid Search with Pinecone
- ⚙️ Powered by LangChain’s `PineconeHybridSearchRetriever`
- 📊 Sparse encoding using BM25 from `pinecone-text`
- 🤖 Embedding model: `all-MiniLM-L6-v2` via HuggingFace
- 🔐 Secure API-based access using `userdata.get()` on Google Colab

---

## 📁 Project Structure

```

.
├── Hybrid\_Search\_RAG.ipynb       # Colab notebook or script
├── bm25\_encoder\_values.json      # Stored TF-IDF values
├── README.md

````

---

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
pip install langchain_community langchain pinecone pinecone-text pinecone-notebooks langchain_huggingface
````

### 2. Configure API Keys

In Google Colab:

```python
from google.colab import userdata
api_key = userdata.get("PINECONE_API_KEY")
hf_token = userdata.get("HF_TOKEN")
```

### 3. Initialize Pinecone Index

```python
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=api_key)

pc.create_index(
    name="hybrid-search-langchain",
    dimension=384,
    metric="dotproduct",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

---

## 🧠 How It Works

### 🔹 Step 1: Embeddings
### 🔸 Step 2: Sparse Encoder (TF-IDF)
### 🔹 Step 3: Hybrid Retriever
### 🔸 Step 4: Add & Retrieve

---

## ✅ What You Learned

* ✅ How to configure and use **Pinecone** for hybrid (dense + sparse) retrieval
* ✅ How to use **BM25Encoder** for sparse (TF-IDF) representation
* ✅ How to integrate **LangChain** with Pinecone for RAG-style applications
* ✅ How to persist sparse encoder values to JSON
* ✅ How hybrid retrieval improves accuracy for simple queries

---

## 🌱 Potential Improvements

* Add a real-world corpus (e.g., FAQs, docs)
* Connect with a Groq or OpenAI LLM for full RAG pipeline
* Integrate into a Streamlit or FastAPI frontend
* Add metadata filters or scoring explanations

---

## 🙌 Credits

* [LangChain](https://www.langchain.com/)
* [Pinecone](https://www.pinecone.io/)
* [Hugging Face](https://huggingface.co/)
* [BM25 Encoder](https://github.com/pinecone-io/pinecone-text)
* [Google Colab](https://colab.research.google.com/)

---

## 🚀 Semantic + Lexical Power Combined with Hybrid Search RAG!
