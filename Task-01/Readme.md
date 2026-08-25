# Prodigy InfoTech Internship - Task 1: Text Generation with Fine-Tuned GPT-2

## 📌 Project Overview
This project implements **Task 1** of the Prodigy InfoTech Internship. The goal is to fine-tune OpenAI's **GPT-2** (a Transformer-based causal language model) on a custom Q&A dataset to generate coherent, stylized, and contextually relevant text based on input prompts.

---

## 🛠️ Tech Stack & Libraries
- **Language**: Python 3
- **Model**: OpenAI GPT-2 (`gpt2`) via Hugging Face `transformers`
- **Frameworks & Libraries**:
  - `PyTorch` (Deep Learning backend)
  - `transformers` (Model architecture & Trainer API)
  - `datasets` (Data management & formatting)
  - `accelerate` (Hardware acceleration)

---

## 🚀 Pipeline Workflow

1. **Hardware Diagnostics**: Automatic GPU (`cuda`) detection with fallback to CPU.
2. **Dataset Preparation**: Curated domain-specific Q&A dataset structured with `<|startoftext|>` and `<|endoftext|>` sequence delimiters.
3. **Tokenization & Preprocessing**: Sequence truncation, max-length padding (`pad_token = eos_token`), and causal LM label matching.
4. **Baseline Benchmark**: Evaluated pre-trained vanilla GPT-2 before fine-tuning.
5. **Fine-Tuning**: Trained using Hugging Face `Trainer` and `DataCollatorForLanguageModeling(mlm=False)` for 8 epochs with AdamW optimization.
6. **Model Serialization**: Exported fine-tuned weights and tokenizer to `./saved_gpt2_custom_model`.
7. **Multi-Strategy Text Generation**:
   - **Greedy Search**
   - **Beam Search** (`num_beams=3`)
   - **Top-K & Top-P Nucleus Sampling** (`temperature=0.7`, `top_p=0.9`)
8. **Quantitative Evaluation**: Measured Evaluation Loss and calculated Perplexity ($\text{PPL} = e^{\text{Loss}}$).

---

## 📊 Results & Output Samples

### Quantitative Metrics
- **Final Training Loss**: Reduced steadily across epochs.
- **Evaluation Perplexity**: Low perplexity indicating high predictive confidence.

### Sample Inference
**Input Prompt**:
```text
Question: What is deep learning?
Answer:
```

**Fine-Tuned GPT-2 Response**:
```text
Deep learning is a specialized subfield of machine learning based on multi-layered artificial neural networks that can automatically extract hierarchical representations from massive datasets.
```

---

## 💻 How to Run

### Google Colab / Jupyter Notebook
1. Open `Prodigy_Task_1_GPT2_Fine_Tuning.ipynb`.
2. Ensure GPU is enabled (**Runtime** > **Change runtime type** > **T4 GPU**).
3. Run all cells sequentially.

