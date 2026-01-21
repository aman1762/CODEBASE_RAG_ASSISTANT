import streamlit as st
import os

from ingest import load_repo, ingest_repo, extract_zip
from vectorstore import create_vectorstore
from rag_chain import build_rag_chain

st.set_page_config(page_title="Codebase RAG Assistant")
st.title("🧠 Codebase RAG Assistant")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

st.subheader("Input Source")

repo_url = st.text_input("GitHub Repository URL (optional)")
uploaded_zip = st.file_uploader("Or upload ZIP file", type=["zip"])

if st.button("Index Codebase"):
    try:
        with st.spinner("Processing codebase..."):

            if uploaded_zip:
                repo_path = extract_zip(uploaded_zip)
                docs = ingest_repo(repo_path)

            elif repo_url:
                if "/blob/" in repo_url:
                    st.error("Please provide a repository URL, not a file URL.")
                    st.stop()
                repo_path = load_repo(repo_url)
                docs = ingest_repo(repo_path)

            else:
                st.warning("Provide GitHub repo URL or upload ZIP.")
                st.stop()

            if not docs:
                st.error(
                    "No supported source files found.\n\n"
                    "Supported: .py, .js, .java, .cpp, .txt, .ipynb"
                )
                st.stop()

            vectorstore = create_vectorstore(docs)
            st.session_state.qa_chain = build_rag_chain(
                vectorstore,
                os.getenv("GROQ_API_KEY")
            )

        st.success("Indexing completed!")

    except Exception as e:
        st.error(str(e))

question = st.text_input("Ask a question about the codebase")

if st.button("Ask"):
    if st.session_state.qa_chain is None:
        st.warning("Index a codebase first.")
    elif not question.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = st.session_state.qa_chain.invoke(question)

        st.markdown("### Answer")
        st.write(answer)
