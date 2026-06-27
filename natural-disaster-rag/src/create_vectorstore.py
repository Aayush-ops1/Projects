import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = "data"
VECTORSTORE_PATH = "vectorstore"

# -----------------------------------
# Load PDFs
# -----------------------------------

all_docs = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(
            os.path.join(DATA_PATH, file)
        )

        docs = loader.load()

        all_docs.extend(docs)

print(f"Loaded {len(all_docs)} pages")

# -----------------------------------
# Chunk Documents
# -----------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(all_docs)

print(f"Created {len(chunks)} chunks")

# -----------------------------------
# Embedding Model
# -----------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------------
# Build FAISS Index
# -----------------------------------

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

# -----------------------------------
# Save Locally
# -----------------------------------

vectorstore.save_local(VECTORSTORE_PATH)

print("Vector Store Saved Successfully")