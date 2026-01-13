# POS Tagging for Extremely Low Resource Indian Languages

## Overview

This repository contains the dataset and code for the research paper titled "POS Tagging for Extremely Low Resource Indian Languages". The objective of this research is to develop and evaluate methods for Part-of-Speech (POS) tagging in extremely low-resource Indian languages using various multilingual massive pretrained language models.


## Files and Directories

- `dataset/`
  - `pos_data_set.xlsx`: Evaluation dataset file.

- `code/`
  - `Baseline (Zero-shot)/`
    - `baseline_zero_shot.py`: Script for zero-shot using three different model MuRIL, XLM-R, RemBERT.
  - `Look-back/`
    - `look_back_py_muril.py`: Script for look-back using MuRIL model.
    - `look_back_xlmr.py`: Script for look-back using XLM-R and Rembert model.
  - `Look-back-with-score/`
    - `look_back_with_score_muril.py`: Script for look-back using MuRIL model.
    - `look_back_with_score_rembert.py`: Script for look-back using RemBERT model.
    - `look_back_with_score_xlmr.py`: Script for look-back using XLM-R model.
  - `Oracle & Non-oracle/`
    - `Oracle_non_oracle_muril.py`: Script for oracle & non-oracle using MuRIL model.
    - `Oracle_non_oracle_muril_rembert.py`: Script for oracle & non-oracle using RemBERT model.
    - `Oracle_non_oracle_muril_xlmr.py`: Script for oracle & non-oracle using XLM-R model.
## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher


## Contact
- For any questions or issues, please contact:
- Sanjeev Kumar: sanjeev@cse.iitb.ac.in

## Bibtext
If you're using the dataset, please cite
@inproceedings{kumar-etal-2024-part,
    title = "Part-of-speech Tagging for Extremely Low-resource {I}ndian Languages",
    author = "Kumar, Sanjeev  and
      Jyothi, Preethi  and
      Bhattacharyya, Pushpak",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.857/",
    doi = "10.18653/v1/2024.findings-acl.857",
    pages = "14422--14431",
    
}
