"""
Core RAG pipeline: chunking → embedding → vector store → retrieval → LLM chain.
Uses 100% free components: HuggingFace Inference API + local embeddings + FAISS.
"""

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

from config import (
    LLM_REPO_ID,
    LLM_TASK,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVER_K,
)


# ─── LLM Setup ───────────────────────────────────────────────────────

def get_llm():
    """
    Create and return a ChatHuggingFace model backed by the free
    HuggingFace Inference API.
    """
    endpoint = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        task=LLM_TASK,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        temperature=LLM_TEMPERATURE,
    )
    return ChatHuggingFace(llm=endpoint)


# ─── Vector Store Construction ────────────────────────────────────────

def build_vectorstore(documents: list):
    """
    Split documents into chunks, embed them, and build a FAISS vector store.

    Args:
        documents: List of LangChain Document objects from the loaders

    Returns:
        FAISS vector store ready for similarity search
    """
    # 1. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)

    # 2. Embed and store in FAISS
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


# ─── RAG Chain ────────────────────────────────────────────────────────

def get_rag_chain(vectorstore):
    """
    Build the full retrieval-augmented generation chain.

    Pipeline:
        User question → Retriever (top-k) → Stuff docs into prompt → LLM → Answer

    Args:
        vectorstore: FAISS vector store with indexed document chunks

    Returns:
        A LangChain retrieval chain that can be invoked with {"input": question}
    """
    # 1. Retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVER_K}
    )

    # 2. Prompt
    system_prompt = (
        "You are a knowledgeable research assistant. "
        "Use ONLY the retrieved context below to answer the user's question. "
        "If the context does not contain enough information to answer, "
        "say 'I don't have enough information from the retrieved documents to answer this.' "
        "Always mention which source (paper title or article name) the information comes from.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 3. Build chains
    llm = get_llm()
    stuff_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, stuff_chain)

    return retrieval_chain


# ─── Query Helper ─────────────────────────────────────────────────────

def ask_question(chain, question: str) -> dict:
    """
    Invoke the RAG chain with a question and return the answer + sources.

    Args:
        chain: The retrieval chain from get_rag_chain()
        question: User's question string

    Returns:
        dict with keys:
            - "answer": The generated answer string
            - "source_documents": List of source Document objects
    """
    response = chain.invoke({"input": question})
    return {
        "answer": response["answer"],
        "source_documents": response.get("context", []),
    }
