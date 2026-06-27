chat_history = []

def add_message(role, content):

    chat_history.append({
        "role": role,
        "content": content
    })


def get_history():

    return chat_history


def format_history():

    history_text = ""

    for msg in chat_history:
        history_text += (
            f"{msg['role']}: "
            f"{msg['content']}\n"
        )

    return history_text