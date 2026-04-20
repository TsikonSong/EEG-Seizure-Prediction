### Reproducibility Guide

To ensure 100% replication of the results presented in our AIIM paper, please follow these steps:

1. **Environment Setup**: 
   Install dependencies via `pip install -r requirements.txt`.

2. **Data Preprocessing**: 
   Run `preprocess_chbmit.py`. 
   *Note: We strictly implemented a **4-hour postictal exclusion gap** to ensure no data leakage and maintain clinical validity.*

3. **Hyperparameters & Manifest**: 
   Specific random seeds and hyperparameter configurations for each of the 24 patient cases are detailed in `patient_specific_results.csv`.

4. **Model Execution**: 
   - The core architectures (TCN, EEGNet, Conformer) are defined in `models.py`.
   - Training workflows are provided in the respective `train_.ipynb` notebooks.

5. **Statistical Analysis**: 
   Refer to `statistical_tests.ipynb` for the p-value calculations and significance testing.
