import os
from question_rewriter import rewrite_question
from chat_memory import (
    add_message,
    format_history
)

from logger import save_log
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --------------------------------
# Load Environment Variables
# --------------------------------

load_dotenv()

# --------------------------------
# Embeddings
# --------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------
# Load FAISS Database
# --------------------------------

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

# --------------------------------
# Groq LLM
# --------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# --------------------------------
# Ask Question Function
# --------------------------------

def ask_question(question):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a Natural Disaster Intelligence Assistant.

Answer ONLY from the provided context.

If the answer is not available in the context,
say:

"I could not find this information in the knowledge base."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, docs
print("\nOriginal Question:", question)
print("Standalone Question:", standalone_question)