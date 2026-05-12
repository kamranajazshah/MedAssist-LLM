# MedAssist-LLM

A domain-specific Medical Large Language Model fine-tuned using QLoRA and Microsoft Phi-2.

## Features

- QLoRA fine-tuning
- 4-bit quantization
- Medical instruction tuning
- Local GPU training
- LoRA adapters
- Inference pipeline

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT
- TRL
- BitsAndBytes
- Accelerate

## Dataset

MedQuad Medical Q&A Dataset:
https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset

## Project Structure

```bash
MedAssist-LLM/
│
├── data/
├── models/
├── src/
│   ├── training/
│   ├── inference/
│   ├── rag/
│   └── api/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Training

```bash
python src/training/train.py
```

## Inference

```bash
python src/inference/chat.py
```

## Future Improvements

- Retrieval-Augmented Generation (RAG)
- FastAPI Backend
- Streamlit Frontend
- Docker Deployment
- Evaluation Pipeline

## Author

Kamran Ajaz Shah