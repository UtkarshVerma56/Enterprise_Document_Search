# Document Search Chatbot

This project is a simple Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions from them.

The application extracts text from PDFs using OCR, stores document embeddings in a vector database, and retrieves the most relevant information when a question is asked. The final response is generated using OpenAI.

I built this project to understand how modern AI-powered document search systems work using tools like LangChain, ChromaDB, Streamlit, and OpenAI APIs.

## How it Works
Upload a policy PDF
Extract text using OCR
Split text into chunks
Convert text into embeddings and store in ChromaDB
Query the system:
Ask questions (RAG mode)
Check compliance of a process

## Tech
Python, Streamlit, LangChain, OpenAI API , OCR

## Features
- Search PDFs by asking questions
- Works with scanned PDFs too
- Shows source documents
