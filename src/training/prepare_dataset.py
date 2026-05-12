from datasets import load_dataset

dataset = load_dataset(
    "keivalya/MedQuad-MedicalQnADataset",
    split="train"
)

def format_example(example):
    text = f"""### Instruction:
{example['Question']}

### Response:
{example['Answer']}
"""
    return {"text": text}

formatted_dataset = dataset.map(format_example)

print(formatted_dataset[0]["text"])