# NeuroBlend

A multimodal deep learning framework for Alzheimer's diagnosis using real MRI, real PET, and synthetic PET.

## What is in this project

- Data preprocessing and registration pipelines for MRI and PET.
- Augmentation pipeline to expand matched MRI-PET pairs.
- MRI-to-synthetic-PET generation models and evaluations.
- Multimodal AD classifier with benchmarking and explainability outputs.

## Project structure

- Classification Module
	- `AD_CLASSIFIER_WITH_EVAL_AND_BENCHMARK.ipynb`
- Data Preprocessing and Augmentation
	- `Preprocessing_Pipeline.ipynb`
	- `Augmentation_Pipeline.ipynb`
- Synthetic PET Generation Module
	- `MRI2PET_SYNTHESIS.ipynb`
	- `MRI2PET_EVALUVATION.ipynb`
	- `MRI2PET_BENCHMARKING.ipynb`

## Quick setup

1. Install Python 3.9+ and Jupyter Lab.
2. Create and activate a virtual environment.
3. Start Jupyter Lab from the project root.

Example (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip jupyterlab notebook ipykernel
jupyter lab
```

Note: Most notebooks include their own dependency-install cells. You can run those cells directly.

## Recommended run order

1. `Data Preprocessing and Augmentation/Preprocessing_Pipeline.ipynb`
2. `Data Preprocessing and Augmentation/Augmentation_Pipeline.ipynb`
3. `Synthetic PET Generation Module/MRI2PET_SYNTHESIS.ipynb`
4. `Synthetic PET Generation Module/MRI2PET_EVALUVATION.ipynb`
5. `Synthetic PET Generation Module/MRI2PET_BENCHMARKING.ipynb`
6. `Classification Module/AD_CLASSIFIER_WITH_EVAL_AND_BENCHMARK.ipynb`

## How to run each notebook

### 1) Preprocessing_Pipeline.ipynb

Purpose:
- Converts and preprocesses MRI/PET data (registration, normalization, resampling, masking).

Before running:
- Update paths in the configuration cell:
	- `INPUT_BASE`
	- `OUTPUT_BASE`
	- `MNI_TEMPLATE_PATH` (optional)

Run:
- Open notebook and run cells top to bottom (`Run All`).

Expected output:
- Preprocessed patient folders with standardized `mri_processed.npy` and `pet_processed.npy` style outputs.

### 2) Augmentation_Pipeline.ipynb

Purpose:
- Creates augmented MRI/PET pairs and combines originals + augmented data.

Before running:
- Update:
	- `DATA_ROOT`
	- `OUT_ROOT`
	- `N_AUG_PER_SUBJECT`

Run:
- Open notebook and run all cells.

Expected output:
- Combined augmented dataset folder (for example, `NACC_Combined_AUG`).

### 3) MRI2PET_SYNTHESIS.ipynb

Purpose:
- Trains the MRI-to-synthetic-PET model.

Before running:
- Update:
	- `DATA_ROOT`
	- `SAVE_PATH`
	- `SAVE_LOSS`
	- `CKPT_PATH`
	- `SAMPLE_DIR`

Run:
- Open notebook and run all cells.

Expected output:
- Trained checkpoints and generated training sample outputs.

### 4) MRI2PET_EVALUVATION.ipynb

Purpose:
- Evaluates a trained synthesis model on validation/test subjects and computes image metrics.

Before running:
- Update:
	- `MODEL_PATH`
	- `EVAL_DATA_ROOT`
	- `EVAL_OUT_DIR`

Run:
- Open notebook and run all cells.

Expected output:
- Quantitative metrics and saved visual comparison results.

### 5) MRI2PET_BENCHMARKING.ipynb

Purpose:
- Benchmarks multiple MRI-to-PET model variants and baselines.

Before running:
- Update:
	- `DATA_ROOT`
	- `CKPT_DIR`

Run:
- Open notebook and run all cells.

Expected output:
- Model comparison metrics, checkpoint files, and benchmark plots/tables.

### 6) AD_CLASSIFIER_WITH_EVAL_AND_BENCHMARK.ipynb

Purpose:
- Trains and evaluates the final AD classifier using multimodal inputs.

Before running:
- Update:
	- `DATA_ROOT`
	- `CSV_PATH`
	- `MODEL_SAVE`
	- `OUT_DIR`

Run:
- Open notebook and run all cells.

Expected output:
- Trained classifier weights, evaluation metrics, confusion matrices, and analysis artifacts.

## Tips

- Use a GPU-enabled environment for training notebooks.
- If memory errors occur, reduce `BATCH_SIZE`, patch size, or number of workers.
- Keep dataset folder naming consistent because notebooks parse subject IDs from folder names.
