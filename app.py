import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings

from document_processor import DocumentProcessor
from config import CHROMA_DB_DIR, EMBEDDING_MODEL

st.set_page_config(page_title="RAG Compliance Bot", page_icon="🛡")
st.title("🛡 RAG + Compliance Checker Bot")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

OPENAI_API_KEY = st.text_input("Enter OpenAI API Key:", type="password")

if uploaded_file and OPENAI_API_KEY:

    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    processor = DocumentProcessor(uploaded_file.name)

    with st.spinner("Extracting text from PDF..."):
        processor.extract_text_from_pdf()

    with st.spinner("Creating vector database..."):
        processor.create_vector_db()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

    client = OpenAI(api_key=OPENAI_API_KEY)

    query = st.text_area("Ask a question from the document:")

    if st.button("Submit"):
        with st.spinner("Searching..."):
            results = db.similarity_search(query, k=3)

            context = "\n\n".join([doc.page_content for doc in results])

            prompt = f"""
            Answer the question based on the context below.

            Context:
            {context}

            Question:
            {query}
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            answer = response.choices[0].message.content

            st.subheader("Answer")
            st.write(answer)
