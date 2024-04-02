import time
import torch
from transformers import GPTJForCausalLM, GPT2Tokenizer
from transformers import AutoTokenizer
import transformers

#export CUDA_VISIBLE_DEVICES=0,1
if torch.cuda.is_available():
    model =  GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16).cuda()
else:
    model =  GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16)


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model.eval()

input_text = "Hello, today I am testing the finetuned GPT J6B model"
input_ids = tokenizer.encode(str(input_text), return_tensors='pt').cuda()

output = model.generate(
    input_ids,
    do_sample=True,
    max_length=1000,
    top_p=0.9,
    top_k=1,
    temperature=2.0,
)
