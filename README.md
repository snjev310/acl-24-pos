# POS Tagging for Extremely Low Resource Indian Languages

## Overview

This repository contains the dataset and code for the research paper titled "POS Tagging for Extremely Low Resource Indian Languages". The objective of this research is to develop and evaluate methods for Part-of-Speech (POS) tagging in extremely low-resource Indian languages using advanced machine learning and natural language processing techniques.


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
- Required Python packages listed in `requirements.txt`

## Contact
- For any questions or issues, please contact:
- Sanjeev Kumar: sanjeev@cse.iitb.ac.in
