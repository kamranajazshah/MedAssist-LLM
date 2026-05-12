#  MedAssist-LLM

A domain-specific Medical Large Language Model fine-tuned using QLoRA and Microsoft Phi-2 for medical question answering and healthcare assistance.

---

#  Project Overview

MedAssist-LLM is an end-to-end medical AI project focused on domain-specific LLM fine-tuning using parameter-efficient training techniques.

The project demonstrates:

- QLoRA Fine-Tuning
- 4-bit Quantization
- LoRA Adapters
- Medical Instruction Tuning
- GPU-Optimized Local Training
- Transformer Inference Pipelines
- CPU/GPU Offloading

The model was fine-tuned locally on an NVIDIA RTX 3050 6GB Laptop GPU using the MedQuad Medical Q&A Dataset.

---

#  Architecture

```text
Medical Dataset
      ↓
Data Formatting
      ↓
QLoRA Fine-Tuning
      ↓
Microsoft Phi-2
      ↓
LoRA Adapters
      ↓
Medical AI Assistant
```

---

#  Features

- ✅ QLoRA Fine-Tuning
- ✅ 4-bit Quantization using BitsAndBytes
- ✅ PEFT LoRA Adapters
- ✅ Medical Instruction Tuning
- ✅ Hugging Face Transformers
- ✅ Local GPU Training
- ✅ Inference Pipeline
- ✅ CPU/GPU Offloading
- ✅ Parameter-Efficient Training

---

#  Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| LLM Framework | Hugging Face Transformers |
| Fine-Tuning | PEFT (LoRA / QLoRA) |
| Training | TRL |
| Quantization | BitsAndBytes |
| GPU Optimization | Accelerate |
| Dataset | MedQuad Medical Q&A Dataset |

---

#  Project Structure

```bash
MedAssist-LLM/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── training/
│   │   ├── load_dataset.py
│   │   ├── prepare_dataset.py
│   │   └── train.py
│   │
│   ├── inference/
│   │   ├── test_model.py
│   │   └── chat.py
│   │
│   ├── rag/
│   │
│   ├── api/
│   │
│   └── utils/
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

#  Dataset

This project uses the MedQuad Medical Question Answering Dataset.

Dataset Link:

https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset

The dataset contains medical question-answer pairs used for instruction tuning.

Example:

```text
Instruction:
What are the symptoms of diabetes?

Response:
Common symptoms include increased thirst, frequent urination, fatigue, blurred vision, and slow healing wounds.
```

---

#  Installation

## 1. Clone Repository

```bash
git clone https://github.com/kamranajazshah/MedAssist-LLM.git

cd MedAssist-LLM
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

#  Fine-Tuning

Run the training pipeline:

```bash
python src/training/train.py
```

The training pipeline includes:

- Dataset Loading
- Dataset Formatting
- QLoRA Fine-Tuning
- 4-bit Quantization
- LoRA Adapter Training

---

# 🧪 Inference

Run inference:

```bash
python src/inference/test_model.py
```

Interactive chatbot:

```bash
python src/inference/chat.py
```

---

#  Training Configuration

| Parameter | Value |
|---|---|
| Base Model | Microsoft Phi-2 |
| Quantization | 4-bit NF4 |
| Fine-Tuning | QLoRA |
| LoRA Rank | 8 |
| Batch Size | 1 |
| Gradient Accumulation | 4 |
| Optimizer | AdamW |
| Precision | FP16 |
| GPU | RTX 3050 6GB |

---

#  Hardware Used

- NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM)
- CUDA 12.8
- Windows Environment

---

#  Engineering Challenges Solved

During development, several real-world LLM engineering challenges were solved:

- CUDA compatibility issues
- BF16 mixed precision errors
- GPU VRAM limitations
- CPU offloading during inference
- Transformers + Accelerate compatibility
- Quantized training configuration
- Windows + PyTorch dependency conflicts

These issues provided hands-on experience with practical LLM infrastructure debugging and optimization.

---

#  What I Learned

Through this project, I gained practical understanding of:

- Transformer Fine-Tuning
- LoRA and QLoRA
- Quantized LLM Training
- GPU Memory Optimization
- Mixed Precision Training
- Hugging Face Ecosystem
- Inference Optimization
- CPU/GPU Offloading
- Instruction Tuning
- LLM Engineering Workflows

---

#  Future Improvements

Planned upgrades for the project:

- Retrieval-Augmented Generation (RAG)
- FAISS Vector Database
- Medical PDF Chat
- FastAPI Backend
- Streamlit Frontend
- Docker Deployment
- Evaluation Pipeline (RAGAS / DeepEval)
- Hugging Face Deployment
- Conversation Memory
- Medical Safety Layer

---

#  Contributions

Contributions, improvements, and suggestions are welcome.

Feel free to fork the repository and open a pull request.

---

#  License

This project is licensed under the MIT License.

---

#  Author

## Kamran Ajaz Shah

- GitHub: https://github.com/kamranajazshah
- LinkedIn: https://www.linkedin.com/in/kamran-shah-070687215

---

#  Acknowledgements

Special thanks to:

- Hugging Face
- Microsoft Phi-2 Team
- PEFT Library
- TRL Library
- BitsAndBytes
- Open-source AI community

---