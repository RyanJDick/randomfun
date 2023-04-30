from pathlib import Path

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.chroma import Chroma


def generate_pdf_embeddings(
    pdf_path: str, embedding_dir: str = "output/embeddings/"
) -> VectorStore:
    """Generate text embeddings for a PDF document.

    Embeddings are persisted under the PDF file name. If embeddings already exist for
    the provided PDF name, they are loaded rather than re-generated.

    Args:
        pdf_path (str): Path to the PDF file to generate embeddings for.
        embedding_dir (str, optional): The directory where embeddings should be
            persisted. Defaults to "output/embeddings/".

    Returns:
        VectorStore: A vector store containing the document embeddings.
    """

    pdf_path = Path(pdf_path)
    embedding_dir = Path(embedding_dir)
    embedding_path = embedding_dir / pdf_path.stem

    # Generate embeddings if they have not already been generated and cached for this
    # PDF file.
    if not embedding_path.exists():
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        sub_docs = text_splitter.split_documents(docs)

        vector_store = Chroma.from_documents(
            sub_docs, OpenAIEmbeddings(), persist_directory=str(embedding_path)
        )

        vector_store.persist()
        print(f"Persisted document embedding to '{embedding_path}'.")
    else:
        print(
            f"Found existing document embeddings at '{embedding_path}'. "
            "Not re-generating."
        )

    # Load persisted embeddings for the PDF file.
    return Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=str(embedding_path),
    )
