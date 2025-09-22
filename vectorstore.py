# vectorstore.py
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document

def prepare_documents(df: pd.DataFrame):
    """
    Convert each row in the DataFrame into a LangChain Document.
    Joins all non-null values in the row into a single string for embedding.
    """
    docs = []
    for i, row in df.iterrows():
        # Combine non-empty values into one text block
        text = " | ".join([str(v) for v in row.dropna().values])
        metadata = {"row_index": i}
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

def build_vectorstore(csv_path: str):
    """
    Build a FAISS vector store from a CSV file using Ollama embeddings.
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Prepare documents for embedding
    docs = prepare_documents(df)

    # Initialize Ollama embeddings (ensure ollama serve is running)
    embeddings = OllamaEmbeddings(model="llama3")

    # Create FAISS vector store
    store = FAISS.from_documents(docs, embeddings)
    return store

def save_to_postgres(df: pd.DataFrame, table_name: str, conn_string: str):
    """
    Demo fallback: Since PostgreSQL isn't installed,
    this saves the DataFrame as a local CSV for now.
    """
    output_file = f"{table_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"[DEMO MODE] Saved locally as {output_file}")

