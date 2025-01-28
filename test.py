from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "Qwen/Qwen2.5-7B-Instruct"
adapter_model = "nnpy/qwen2-7b-chat-ft" # replace with your adapter model

model = AutoModelForCausalLM.from_pretrained(base_model, device_map='auto')
model = PeftModel.from_pretrained(model, adapter_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)

model.eval()

prompt = "hey whatup!"

tokenized = tokenizer.apply_chat_template(
    [
        {
            'role': 'user',
            'content': 'hey love you babe'
        }
    ],
    tokenize=False
)

tokens = tokenizer(tokenized, return_tensors='pt').to('cuda')

res = model.generate(**tokens, max_new_tokens=100)
print(tokenizer.decode(res.reshape(-1)))