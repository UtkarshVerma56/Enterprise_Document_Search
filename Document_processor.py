%%writefile rag_compliance_app.py
import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from compliance_checker import ComplianceChecker

st.set_page_config(page_title="RAG Compliance Bot", page_icon="🛡")
st.title("🛡 RAG + Compliance Checker Bot")

OPENAI_API_KEY = st.text_input("Enter OpenAI API Key:", type="password")
if not OPENAI_API_KEY:
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# Load DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

mode = st.radio(
    "Select Mode:",
    ["Normal Q&A (RAG)", "Compliance Checker"],
    horizontal=True
)

query = st.text_area("Enter your query or business process description:")

if st.button("Submit"):
    with st.spinner("Processing..."):

        if mode == "Compliance Checker":
            checker = ComplianceChecker(db, embeddings, OPENAI_API_KEY)
            result = checker.evaluate(query)

            st.subheader("🛡 Compliance Result (JSON)")
            st.json(result)

            st.subheader("📄 Policy Evidence Used")
            st.text(result["evidence"])

        else:
            # Normal RAG Flow
            results = db.similarity_search(query, k=3)
            context = "\n\n".join([d.page_content for d in results])

            prompt = f"""
Use ONLY the context to answer.

Context:
{context}

Question: {query}
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            st.subheader("Answer")
            st.write(response.choices[0].message.content)
