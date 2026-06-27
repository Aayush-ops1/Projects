from rag_chain import ask_question

print("\nNatural Disaster Intelligence Assistant")
print("Type 'exit' to quit\n")

while True:

    question = input("Question: ")

    if question.lower() == "exit":
        break

    answer, docs = ask_question(question)

    print("\nAnswer:\n")
    print(answer)

    print("\nSources:")

    for doc in docs:
        print(
            f"{doc.metadata.get('source')} "
            f"(Page {doc.metadata.get('page')})"
        )

    print("\n" + "=" * 80 + "\n")