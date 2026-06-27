import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

print("API KEY FOUND:", os.getenv("GROQ_API_KEY"))

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

def rewrite_question(
    question,
    history
):

    prompt = f"""
Given the conversation history,
rewrite the user's latest question
as a standalone question.

Conversation:
{history}

Question:
{question}

Standalone Question:
"""

    response = llm.invoke(prompt)

    return response.content.strip()