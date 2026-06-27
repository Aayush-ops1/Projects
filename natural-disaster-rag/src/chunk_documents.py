from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

DATA_PATH = "data"

all_docs = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(
            os.path.join(DATA_PATH, file)
        )

        docs = loader.load()

        all_docs.extend(docs)

print(f"Loaded {len(all_docs)} pages")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(all_docs)

print(f"Created {len(chunks)} chunks")

print("\nSample Chunk:\n")
print(chunks[0].page_content[:500])