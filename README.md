
## Overview
# Part-of-Speech Tagging for Extremely Low-Resource Indian Languages

This repository contains the **dataset, code, and experimental setup** for the paper:

> **Part-of-Speech Tagging for Extremely Low-resource Indian Languages**  
> *Findings of the Association for Computational Linguistics (ACL) 2024*

---

## 1. Introduction

Part-of-Speech (POS) tagging is a fundamental task in Natural Language Processing (NLP). However, for **extremely low-resource Indian languages**, the lack of annotated corpora and linguistic tools makes POS tagging particularly challenging.

This work explores **zero-shot and weakly supervised POS tagging methods** for extremely low-resource Indian languages using **multilingual pre-trained language models (PLMs)**. We introduce and analyze simple yet effective inference-time strategies—such as **look-back**, **look-back-with-score**, and **oracle/non-oracle variants**—that improve POS tagging performance **without requiring additional labeled data**.

This repository provides:
- The **evaluation dataset**
- Complete **code for all baselines and proposed methods**
- Scripts to reproduce experiments reported in the paper

---

## 2. Languages Covered

The experiments focus on **extremely low-resource Indian languages**, primarily from the Indo-Aryan family, including:

- Angika  
- Magahi  
- Bhojpuri  

All experiments are conducted in **zero-shot or minimally supervised settings**.

---

## 3. Repository Structure

### Dataset
- `dataset/`
  - `pos_data_set.xlsx` — Evaluation dataset

### Code
- `code/`

  **Baseline (Zero-shot)**
  - `baseline_zero_shot.py` — Zero-shot POS tagging using MuRIL, XLM-R, and RemBERT

  **Look-back**
  - `look_back_py_muril.py` — Look-back method using MuRIL
  - `look_back_xlmr.py` — Look-back method using XLM-R and RemBERT

  **Look-back-with-score**
  - `look_back_with_score_muril.py` — MuRIL
  - `look_back_with_score_rembert.py` — RemBERT
  - `look_back_with_score_xlmr.py` — XLM-R

  **Oracle & Non-oracle**
  - `Oracle_non_oracle_muril.py` — MuRIL
  - `Oracle_non_oracle_muril_rembert.py` — RemBERT
  - `Oracle_non_oracle_muril_xlmr.py` — XLM-R

## 4. Dataset

### 4.1 File Description

- **`dataset/pos_data_set.xlsx`**

This file contains the **evaluation dataset** used in all experiments.  
It includes:
- Sentences in extremely low-resource Indian languages
- Token-level annotations
- Gold POS tags

⚠️ **Note:**  
The dataset is intended **only for evaluation and analysis**, not for supervised training.

---

## 5. Models Used

The following **multilingual pre-trained language models** are used in this work:

- **MuRIL**
- **XLM-R (XLM-RoBERTa)**
- **RemBERT**

All models are used **without fine-tuning on the target languages**, strictly in a **zero-shot inference setting**.

---


## 6. Methods Implemented

### 6.1 Baseline (Zero-shot)

**Directory:** `code/Baseline (Zero-shot)/`

- `baseline_zero_shot.py`

Implements standard zero-shot POS tagging using masked language modeling probabilities from multilingual PLMs, without additional heuristics.

---

### 6.2 Look-back Method

**Directory:** `code/Look-back/`

The look-back method revisits previously predicted tokens to refine POS tag assignments based on contextual consistency.

Scripts:
- `look_back_py_muril.py` — MuRIL
- `look_back_xlmr.py` — XLM-R and RemBERT

---

### 6.3 Look-back-with-score

**Directory:** `code/Look-back-with-score/`

Extends the look-back method by incorporating **model confidence scores** while revising POS predictions.

Scripts:
- `look_back_with_score_muril.py`
- `look_back_with_score_rembert.py`
- `look_back_with_score_xlmr.py`

---

### 6.4 Oracle and Non-oracle Experiments

**Directory:** `code/Oracle & Non-oracle/`

These experiments analyze upper bounds and realistic scenarios by controlling access to gold versus predicted tags.

Scripts:
- `Oracle_non_oracle_muril.py`
- `Oracle_non_oracle_muril_rembert.py`
- `Oracle_non_oracle_muril_xlmr.py`

---

## 7. Running the Experiments

### 7.1 Prerequisites

- Python **3.7 or higher**
- PyTorch
- Hugging Face `transformers`
- Pandas
- NumPy

## 8. 🚀 Getting Started
You can load the dataset directly using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# Load the test split
dataset = load_dataset("snjev310/bihari-languages-upos", split="test")

# Access the first sentence in Angika
print(f"Tokens: {dataset[0]['angika_token']}")
print(f"UPOS IDs: {dataset[0]['angika_upos']}")

# Map integer IDs back to tag names
labels = dataset.features["angika_upos"].feature.names
readable_tags = [labels[i] for i in dataset[0]['angika_upos']]
print(f"UPOS Tags: {readable_tags}")
```
## Citation

If you use the dataset or code from this repository, please cite the following paper:

```bibtex
@inproceedings{kumar-etal-2024-part,
    title     = {Part-of-Speech Tagging for Extremely Low-resource Indian Languages},
    author    = {Kumar, Sanjeev and
                 Jyothi, Preethi and
                 Bhattacharyya, Pushpak},
    booktitle = {Findings of the Association for Computational Linguistics: ACL 2024},
    month     = aug,
    year      = {2024},
    address   = {Bangkok, Thailand},
    publisher = {Association for Computational Linguistics},
    url       = {https://aclanthology.org/2024.findings-acl.857/},
    doi       = {10.18653/v1/2024.findings-acl.857},
    pages     = {14422--14431}
}
```
## Contact
- For any questions or issues, please contact:
- Sanjeev Kumar: sanjeev@cse.iitb.ac.in
