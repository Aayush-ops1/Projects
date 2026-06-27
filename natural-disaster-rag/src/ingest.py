from langchain_community.document_loaders import PyPDFLoader
import os

DATA_PATH = "data"

all_docs = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(DATA_PATH, file)

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        all_docs.extend(docs)

print(f"Loaded {len(all_docs)} pages")