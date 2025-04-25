import csv
import os
from typing import List

from ...data_vectorizer import get_context_chunks
from food_cutting_qa.prompting.gemma_prompter import GemmaPrompter
from food_cutting_qa.prompting.llama_prompter import LlamaPrompter
from food_cutting_qa.prompting.prompter import Prompter


def read_questions() -> List[str]:
    questions = []
    data_path = os.path.join(os.path.dirname(__file__), "..", "qa.csv")
    with open(data_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            questions.append(row[0])
    return questions


def prompt_models(questions: List[str], models: List[Prompter], dbs: List[str], context_amount=3):
    system_msg = "Please answer the following question as briefly and concise as you can, using the provided context as help."
    result_path = os.path.join(os.path.dirname(__file__), "..", "model_answers")
    for prompter in models:
        for db in dbs:
            prompt_res = []
            for q in questions:
                context = get_context_chunks(db, q, context_amount)
                user_msg = f'Question: {q}\nContext: {context}\nAnswer:'
                res = prompter.prompt_model(system_msg, user_msg)
                prompt_res.append((q, res, context))
            with open(f"{result_path}/{prompter.model_name.lower()}_{db}.csv", "w", newline="",
                      encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["question", "model_answer", "context"])
                writer.writerows(prompt_res)


if __name__ == "__main__":
    prompters = [LlamaPrompter(), GemmaPrompter()]
    databases = ['recipes', 'wikihow', 'tutorials']
    questions = read_questions()
    prompt_models(questions, prompters, databases)
