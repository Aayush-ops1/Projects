import json
import os
from datetime import datetime

LOG_FILE = "logs/chat_logs.json"


def save_log(question,
             answer,
             sources):

    log_entry = {

    "timestamp": ...,

    "question": question,

    "rewritten_question":
    standalone_question,

    "answer": answer,

    "sources": sources
}

    os.makedirs("logs",
                exist_ok=True)

    logs = []

    if os.path.exists(LOG_FILE):

        with open(
            LOG_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            try:
                logs = json.load(f)

            except:
                logs = []

    logs.append(log_entry)

    with open(
        LOG_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            logs,
            f,
            indent=4,
            ensure_ascii=False
        )