from datasets import load_dataset

dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")

print(dataset[0])
print(f"Dataset Size: {len(dataset)}")