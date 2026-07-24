# Groq LLM Demo

This program uses the Groq LLM to answer several questions, including one question that uses Retrieval-Augmented Generation (RAG).  
The script loads a text file, builds a FAISS vector store, retrieves relevant text chunks, and uses Groq to answer a question using that retrieved context.
---

##  What This Project Does

- Uses Groq’s LLaMA model to answer:
  - Normal questions
  - One **RAG question** using retrieved context

