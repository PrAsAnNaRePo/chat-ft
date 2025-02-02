from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="HF Model ID")
parser.add_argument("--adapter_model", type=str, required=True, help="Path to the adapter model")
parser.add_argument("--prompt", type=str, required=True, help="Prompt to use for generation")

args = parser.parse_args()

base_model = args.model_id
adapter_model = args.adapter_model
prompt = args.prompt

model = AutoModelForCausalLM.from_pretrained(base_model, device_map='auto')
model = PeftModel.from_pretrained(model, adapter_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)

model.eval()

tokenized = tokenizer.apply_chat_template(
    [
        {
            'role': 'user',
            'content': prompt
        }
    ],
    tokenize=False
)

tokens = tokenizer(tokenized, return_tensors='pt').to('cuda')

res = model.generate(**tokens, max_new_tokens=150)
print(tokenizer.decode(res.reshape(-1)))