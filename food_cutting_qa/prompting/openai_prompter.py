import json
import os

from openai import OpenAI

from food_cutting_qa.prompting.prompter import Prompter


class OpenAIPrompter(Prompter):
    def __init__(self):
        super().__init__("gpt-4o-2024-11-20")
        credentials = os.path.join(os.path.dirname(__file__), "..", "prompting", "credentials.json")
        json_text = json.load(open(credentials))
        self.client = OpenAI(
            api_key=json_text["api_key"],
        )

    def prompt_model(self, system_msg: str, question: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": question},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content
