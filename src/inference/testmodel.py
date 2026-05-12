import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(base_model)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload",
    offload_state_dict=True,
)

model = PeftModel.from_pretrained(
    model,
    "./models/medical-phi2",
    device_map="auto",
    offload_folder="offload",
)

model.config.use_cache = True


prompt = """### Instruction:
What are the symptoms of cholera?

### Response:
"""



inputs = tokenizer(prompt, return_tensors="pt")

# Move tensors properly
inputs = {k: v.to(model.device) for k, v in inputs.items()}


with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n===== MODEL RESPONSE =====\n")
print(response)