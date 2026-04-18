from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

FAISS_INDEX_PATH = "faiss_index"


def build_vectorstore(pdf_path: str) -> FAISS:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY")
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore


def load_vectorstore() -> FAISS:
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY")
    )
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def get_answer(question: str, vectorstore: FAISS) -> dict:
    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        api_key=os.getenv("MISTRAL_API_KEY")
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided document.
Use only the context below to answer. If the answer is not in the context, say
"I couldn't find that information in the document."

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)

    source_docs = retriever.invoke(question)
    sources = []
    for doc in source_docs:
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content[:200].replace("\n", " ").strip()
        sources.append({"page": page + 1, "snippet": snippet})

    return {"answer": answer, "sources": sources}


def index_exists() -> bool:
    return os.path.exists(f"{FAISS_INDEX_PATH}/index.faiss")
