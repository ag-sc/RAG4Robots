import csv
import glob
import os
import re
from pathlib import Path

result_path = os.path.join(os.path.dirname(__file__), "..", "results")
file_ext_bin = "binary"
file_ext_score = "scoring"


def get_binary_from_text(text: str) -> bool:
    cont_true = cont_false = False
    if "true" in text.lower():
        cont_true = True
    if "false" in text.lower():
        cont_false = True
    assert cont_true != cont_false
    return cont_true


def get_score_from_text(text: str) -> int:
    match = re.search(r'\b\d+\b', text)
    if match is None:
        return -1
    else:
        return int(match.group())

def evaluate_binary_results():
    for fname in glob.glob(f"{result_path}/*{file_ext_bin}.csv"):
        correct = 0
        rows = 0
        with open(f"{fname}", "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                rows += 1
                if get_binary_from_text(row[3]):
                    correct += 1
            accuracy = correct / rows
        path = Path(fname)
        print(f'For {path.stem}: {correct} answers are correct. Accuracy = {accuracy}\n')


def evaluate_scoring_results():
    for fname in glob.glob(f"{result_path}/*{file_ext_score}.csv"):
        total = 0
        rows = 0
        no_score = 0
        with open(f"{fname}", "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                score = get_score_from_text(row[3])
                if score == -1:
                    no_score += 1
                else:
                    rows += 1
                    total += score
            avg = total / rows
        path = Path(fname)
        print(f'For {path.stem}, the average score is {avg}. However, {no_score} answers were not given a score by the evaluating LLM\n')


if __name__ == "__main__":
    evaluate_binary_results()
    evaluate_scoring_results()
