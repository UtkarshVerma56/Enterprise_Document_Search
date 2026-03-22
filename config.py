from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load text
loader = TextLoader("output.txt", encoding="utf-8")
documents = loader.load()

# Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)
print("Chunks:", len(docs))

# Embedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create ChromaDB
db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
print("ChromaDB created!")
