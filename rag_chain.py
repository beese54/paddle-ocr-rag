import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "paddle_ocr_docs"


def build_chain():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Reformulates the user question as a standalone question, accounting for chat history
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question which might reference "
         "context in the chat history, formulate a standalone question that can be "
         "understood without the chat history. Do NOT answer the question — just "
         "reformulate it if needed, otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # Answers the question using retrieved context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant that answers questions about a scanned technical document. "
         "Use the retrieved context below to answer. The context may include text from the "
         "document as well as descriptions of figures, diagrams, and images. "
         "If you don't know the answer, say so clearly. Keep answers concise and accurate.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain
