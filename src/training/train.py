import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)

from peft import LoraConfig
from trl import SFTTrainer

# ==========================================
# MODEL NAME
# ==========================================

model_name = "microsoft/phi-2"

# ==========================================
# LOAD DATASET
# ==========================================

dataset = load_dataset(
    "keivalya/MedQuad-MedicalQnADataset",
    split="train[:1000]"
)

# ==========================================
# FORMAT DATASET
# ==========================================

def format_example(example):
    text = f"""### Instruction:
{example['Question']}

### Response:
{example['Answer']}
"""
    return {"text": text}

dataset = dataset.map(format_example)

# ==========================================
# QUANTIZATION CONFIG
# ==========================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ==========================================
# TOKENIZER
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# LOAD MODEL
# ==========================================

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# IMPORTANT
model.config.use_cache = False

# ==========================================
# LORA CONFIG
# ==========================================

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["Wqkv", "fc1", "fc2"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ==========================================
# TRAINING ARGUMENTS
# ==========================================

training_args = TrainingArguments(
    output_dir="./models",

    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    learning_rate=2e-4,
    num_train_epochs=1,

    logging_steps=10,
    save_strategy="epoch",

    # DISABLE MIXED PRECISION
    fp16=False,
    bf16=False,

    optim="adamw_torch",

    report_to="none",
)

# ==========================================
# TRAINER
# ==========================================

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
)

# ==========================================
# TRAIN
# ==========================================

trainer.train()

# ==========================================
# SAVE MODEL
# ==========================================

trainer.model.save_pretrained("./models/medical-phi2")

tokenizer.save_pretrained("./models/medical-phi2")

print("Training Complete!")