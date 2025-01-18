# PCA and Logistic Regression as a Diagnostic Tool for Alzheimer's Disease

## Description
This repository contains the data, codes, and files associated with the published scientific article:

> **PCA and logistic regression in 2-[18F]FDG PET neuroimaging as an interpretable and diagnostic tool for Alzheimer's disease.**
>
> Published in: *Physics in Medicine & Biology, 2024* ([DOI: 10.1088/1361-6560/ad0ddd](https://doi.org/10.1088/1361-6560/ad0ddd))

The research proposes and implements an optimization and training pipeline for classifying PET-FDG images aimed at diagnosing Alzheimer's Disease (AD). The model combines Principal Component Analysis (PCA) and Logistic Regression (LR) to offer an interpretable and efficient solution.

---

## Repository Structure

- **`.streamlit/`**: Streamlit application configurations.
- **`data/`**: Contains neuroimaging data:
  - **`ADNI_AD/`**: PET images of AD patients from ADNI.
  - **`ADNI_CN/`**: PET images of cognitively normal individuals from ADNI.
  - **`CDI_AD/`**: PET images of AD patients from CDI.
  - **`CDI_nAD/`**: PET images of individuals without evidence of AD from CDI.
  - **`ADNI_dataset_info.csv`** and **`CDI_dataset_info.csv`**: Demographic information (gender, age, etc.) of the patients.
- **`model/`**: Contains the trained model files, including logistic regression coefficients and PCA parameters.
- **`paper_code_and_files/`**: Code to reproduce the article results (Jupyter Notebook).
- **`requirements.txt`**: Dependencies needed to reproduce the project.
- **`webapp.py`**: Source code for the web application built with Streamlit.
- **`generate_lighter_files.py`**: Script to generate lighter model files.

---

## Data

The data used in this project were obtained from two sources:

1. **Alzheimer's Disease Neuroimaging Initiative (ADNI):**
   - 100 PET images of AD patients.
   - 100 PET images of cognitively normal individuals (CN).
2. **Centro de Diagnóstico por Imagem (CDI):**
   - 92 PET images of AD patients.
   - 100 PET images of individuals without evidence of AD.

The images were preprocessed using SPM12 to ensure reorientation, spatial normalization (MNI space), and smoothing.

---

## Web Application

The Streamlit application provides:

1. **Report Generation:** Personalized AD prediction based on input PET images.
2. **Neuroimage Exploration:** Interactive visualization of PET images across different planes.
3. **Model Understanding:** Educational insights into the PCA and LR techniques used in the model.

You can access the application in this link: [https://pca-lr-ad-prediction.streamlit.app/](https://pca-lr-ad-prediction.streamlit.app/)

### Running the Application Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/cego669/PCA-and-LR-as-a-diagnostic-tool-for-AD.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run webapp.py
   ```

Access it in your browser at `http://localhost:8501`.

---

## Reproducing the Results

The results of the article can be reproduced using the notebook available in `paper_code_and_files/code_to_reproduce_results.ipynb`. It includes:

- Hyperparameter optimization.
- Model training with ADNI data.
- Testing with CDI data.

### Requirements

Ensure that the dependencies listed in `requirements.txt` are installed before running the notebook.

---

## Citation

Please cite this work as follows:

```
Gonçalves de Oliveira CE, Araújo WM, Teixeira ABMJ, Gonçalves GL,
Itikawa EN. PCA and logistic regression in 2-[18F]FDG PET neuroimaging as
an interpretable and diagnostic tool for Alzheimer's disease. Phys Med Biol.
2024. doi: 10.1088/1361-6560/ad0ddd
```

---

## Contact

For questions or comments, contact:
- Carlos Eduardo Gonçalves de Oliveira: [LinkedIn](https://www.linkedin.com/in/cego669/) | [GitHub](https://github.com/cego669)