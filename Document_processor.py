from pdf2image import convert_from_path
import pytesseract
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class DocumentProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.output_text_file = "output.txt"
        self.persist_directory = "./chroma_db"

    def extract_text_from_pdf(self):
        pages = convert_from_path(self.pdf_path)

        text = ""
        for i, page in enumerate(pages):
            extracted = pytesseract.image_to_string(page)
            text += f"\n\n--- PAGE {i+1} ---\n\n{extracted}"

        with open(self.output_text_file, "w", encoding="utf-8") as f:
            f.write(text)

        print("Extracted preview:\n", text[:1000])
        return self.output_text_file

    def create_vector_db(self):
        loader = TextLoader(self.output_text_file, encoding="utf-8")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = splitter.split_documents(documents)

        print("Chunks:", len(docs))

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=self.persist_directory
        )

        print("ChromaDB created!")
        return db
