!apt install poppler-utils tesseract-ocr -y
!pip install streamlit openai pdf2image pytesseract langchain_community chromadb sentence-transformers
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb

from google.colab import files
uploaded = files.upload()   # Upload your policy PDF
pdf_path = list(uploaded.keys())[0]

from pdf2image import convert_from_path
import pytesseract

pages = convert_from_path(pdf_path)

text = ""
for i, page in enumerate(pages):
    extracted = pytesseract.image_to_string(page)
    text += f"\n\n--- PAGE {i+1} ---\n\n{extracted}"

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Extracted preview:\n", text[:1000])
