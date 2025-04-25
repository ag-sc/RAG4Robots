import re

import torch
import transformers

from food_cutting_qa.prompting.prompter import Prompter


class GemmaPrompter(Prompter):
    def __init__(self):
        super().__init__("gemma-2-27b-it")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(f'google/{self.model_name}')
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            f'google/{self.model_name}',
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

    def prompt_model(self, system_msg: str, question: str) -> str:
        messages = [{"role": "user", "content": f"{system_msg}\n{question}"}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs.to(self.model.device),
                                      max_new_tokens=self.max_new_tokens,
                                      do_sample=False,  # No randomness if False
                                      temperature=None,
                                      top_p=None)
        outputs = self.tokenizer.decode(outputs[0, len(inputs):])
        match = re.search(r"<start_of_turn>model(.*?)<end_of_turn>", outputs, re.DOTALL)
        result = match.group(1).strip() if match else None
        return result
