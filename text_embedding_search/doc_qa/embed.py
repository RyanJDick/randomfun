from pathlib import Path

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores.chroma import Chroma


def generate_pdf_embeddings(
    pdf_path: str, embedding_dir: str = "output/embeddings/"
) -> VectorStoreIndexWrapper:
    """Generate text embeddings for PDF document.

    Embeddings are persisted under the PDF file name. If embeddings already exist for
    the provided PDF name, they are loaded rather than re-generated.

    Args:
        pdf_path (str): Path to the PDF file to generate embeddings for.
        embedding_dir (str, optional): The directory where embeddings should be
            persisted. Defaults to "output/embeddings/".

    Returns:
        VectorStoreIndexWrapper: A vector store containing the document embeddings.
    """
    loader = PyPDFLoader(pdf_path)

    pdf_path = Path(pdf_path)
    embedding_dir = Path(embedding_dir)
    embedding_path = embedding_dir / pdf_path.stem

    # Generate embeddings if they have not already been generated and cached for this
    # PDF file.
    if not embedding_path.exists():
        index = VectorstoreIndexCreator(
            embedding=OpenAIEmbeddings(),
            vectorstore_cls=Chroma,
            vectorstore_kwargs={"persist_directory": str(embedding_path)},
        ).from_loaders([loader])

        index.vectorstore.persist()
        print(f"Persisted document embedding to '{embedding_path}'.")
    else:
        print(
            f"Found existing document embeddings at '{embedding_path}'. "
            "Not re-generating."
        )

    # Load persisted embeddings for the PDF file.
    # Note: The langchain API is a little weird here. Index construction looks very
    #       different when you want to load from a persistent store.
    return VectorStoreIndexWrapper(
        vectorstore=Chroma(
            embedding_function=OpenAIEmbeddings(),
            persist_directory=str(embedding_path),
        )
    )
