from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def build_rag_chain(vectorstore, groq_api_key):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are a codebase assistant.

        RULES:
        - Use ONLY the context below.
        - If information is missing, say:
          "I cannot find this information in the provided codebase."
        - Do NOT guess.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    def format_docs(docs):
        return "\n\n".join(
            f"FILE: {d.metadata['file']}\n{d.page_content}"
            for d in docs
        )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
