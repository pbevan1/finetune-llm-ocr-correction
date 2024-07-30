import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from cog import BasePredictor, Input


class Predictor(BasePredictor):
    def setup(self):
        model_id = "pbevan11/llama-3.1-8b-ocr-correction-merged"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict(
        self,
        instruction: str = Input(description="Instruction for the model"),
        inp: str = Input(description="Input text to correct"),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate", default=5000
        ),
        temperature: float = Input(
            description="Temperature for sampling", default=1.0, ge=0.0, le=2.0
        ),
        top_p: float = Input(description="Top-p sampling", default=1.0, ge=0.0, le=1.0),
        top_k: int = Input(description="Top-k sampling", default=50, ge=0),
        repetition_penalty: float = Input(
            description="Repetition penalty", default=1.0, ge=0.0
        ),
        do_sample: bool = Input(description="Whether to use sampling", default=False),
    ) -> str:
        prompt = self.create_prompt(instruction, inp)
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).input_ids.cuda()

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
        }

        out_ids = self.model.generate(input_ids=input_ids, **generation_config)
        full_output = self.tokenizer.batch_decode(
            out_ids.detach().cpu().numpy(), skip_special_tokens=True
        )[0]
        response_start = full_output.find("### Response:")
        if response_start != -1:
            response = full_output[response_start + len("### Response:") :]
        else:
            response = full_output[len(prompt) :]

        # Remove backslashes that aren't part of a newline
        response = re.sub(r"\\(?!n)", "", response)

        return response

    def create_prompt(self, instruction, inp):
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{inp}

### Response:
"""
