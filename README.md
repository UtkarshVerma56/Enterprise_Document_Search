# Document Search Chatbot

Chatbot that searches your company PDFs (HR, Finance, Technical).

## How it Works
Upload a policy PDF
Extract text using OCR
Split text into chunks
Convert text into embeddings and store in ChromaDB
Query the system:
Ask questions (RAG mode)
Check compliance of a process

## Tech
Python, Streamlit, LangChain, OpenAI, FAISS, OCR

## Features
- Search PDFs by asking questions
- Works with scanned PDFs too
- Shows source documents
