!apt install poppler-utils tesseract-ocr -y
!pip install streamlit openai pdf2image pytesseract langchain_community chromadb sentence-transformers
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb
