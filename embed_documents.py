import os
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import pickle

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Where to save FAISS stores
output_dir = "vectorstores"
os.makedirs(output_dir, exist_ok=True)

# Domains
domains = ["finance", "marketing", "general"]

for domain in domains:
    docs = []
    folder_path = f"data/{domain}"
    print(f"🔍 Loading documents from: {folder_path}")

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, filename))
            loaded_docs = loader.load()
            docs.extend(loaded_docs)

    # Split long documents into smaller chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    # Create vector store
    print(f"📦 Embedding and saving FAISS vector store for: {domain}")
    vectordb = FAISS.from_documents(split_docs, embedding_model)
    
    domain_store_path = os.path.join(output_dir, domain)
    os.makedirs(domain_store_path, exist_ok=True)
    vectordb.save_local(domain_store_path)

print("\n✅ All vector stores created and saved under vectorstores/")
