# AVA-Qwen1.5-7
Fine-Tuned Qwen1.5 7B Persian Large Language Model LLM / Persian Qwen1.5 7B



# AVA-Qwen1.5 / Persian Qwen 


 <img src="https://github.com/mehdihosseinimoghadam/AVA-Qwen1.5-7/blob/main/AVA-Qwen.png" height="600" width="940" >

### This Repository Contains Documents for Fine-Tuned Qwen1.5  Persian Large Language Model(LLM) Called AVA-Qwen1.5
(Still in progress)

-------------------------------------------------
### Dataset used:

To Be Done

-------------------------------------------------



### Usage:

All models are hosted in HuggingFace, and here is the code for inference:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

model_name_or_id = "MehdiHosseiniMoghadam/AVA-Qwen1.5-7B-Chat"
model = AutoModelForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True, load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_id)

prompt = ''

prompt = f"### Human:{prompt}\n### Assistant:"


inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

generation_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    temperature=0.01,
    max_new_tokens=90,
    pad_token_id=tokenizer.eos_token_id
)


outputs = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


```

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

Released Jan 30, 2024 by [Mehdi Hosseini Moghadam](https://github.com/mehdihosseinimoghadam)

Attention ⚠️: The user is responsible for using AVA-Llama-3 / Persian Llama 3

Any misuse of the model (of any kind) is the responsibility of the user and not the creator


## Contact

<a href="https://ir.linkedin.com/in/mehdi-hosseini-moghadam-384912198" target="_blank"><img src="https://cdn-icons.flaticon.com/png/512/3536/premium/3536505.png?token=exp=1644871115~hmac=59bc0b44906adebd63f84642086d4695" alt="Buy Me A Coffee" style="height: 50px !important;width: 50px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>
	
	
<a href="https://scholar.google.com/citations?user=TKWbohsAAAAJ&hl=en" target="_blank"><img src="https://cdn-icons.flaticon.com/png/512/3107/premium/3107171.png?token=exp=1644871560~hmac=7f8fd85e8db71945e25202a3ac739e1c" alt="Buy Me A Coffee" style="height: 50px !important;width: 50px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

<a href="https://huggingface.co/MehdiHosseiniMoghadam" target="_blank"><img src="https://cdn-icons.flaticon.com/png/512/2461/premium/2461892.png?token=exp=1644871873~hmac=8659d04d69008e399a5344cad5bc4270" alt="Buy Me A Coffee" style="height: 50px !important;width: 50px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>





 
