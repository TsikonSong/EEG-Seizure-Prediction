To replicate the results in our AIIM paper:

Data Preprocessing: Run preprocess_chbmit.py. Note that we implemented a 4-hour postictal exclusion gap to ensure no data leakage between seizures.

Hyperparameters: Detailed random seeds and hyperparameter settings for each of the 24 cases are documented in patient_specific_results.json.

Model Training: The main architectures (TCN, EEGNet, etc.) are defined in models.py. Training procedures can be found in the provided notebooks.
