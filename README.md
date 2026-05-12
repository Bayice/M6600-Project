# Robust Accent Classification with Speaker-Disjoint Evaluation and Semi-Supervised Data Scaling

Author: He Xu  
UNI: xh2707  
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control  
Instructor: Prof. Homayoon Beigi  

## 1. Project Overview

This project studies English accent classification using DistilHuBERT-based audio classifiers. The original reference model, `kaysrubio/accent-id-distilhubert-finetuned-l2-arctic2`, supports six first-language accent classes from the L2-ARCTIC corpus:

- Arabic
- Hindi
- Korean
- Mandarin
- Spanish
- Vietnamese

The project investigates whether high accuracy on the original L2-ARCTIC dataset truly reflects robust accent generalization. The main experimental design compares file-level evaluation, speaker-disjoint evaluation, external data expansion, balanced external training, and semi-supervised pseudo-labeling.

The main finding is that file-level evaluation can substantially overestimate model robustness. Speaker-disjoint evaluation is more realistic because speakers in the test set are not present in the training set. External-only training does not transfer well back to L2-ARCTIC, while combining L2-ARCTIC with carefully balanced external accent data is more promising.

## 2. Main Contributions

This project includes the following components:

1. Reproduction of a file-level L2-ARCTIC accent classification baseline.
2. Speaker-disjoint four-fold evaluation on the official L2-ARCTIC corpus.
3. Evaluation of the public Hugging Face accent classifier on internal and external samples.
4. Construction of external English accent metadata from multiple public datasets.
5. Round1 diagnostic experiments comparing L2-only, external-only, and L2-plus-external training.
6. Round2 balanced training with British Isles English and Singaporean English as expanded labels.
7. Semi-supervised teacher-student pseudo-labeling using high-confidence external samples.
8. A second-level external evaluation set for cross-corpus generalization testing.

## 3. Repository Structure

```text
M6600_Project/
├── configs/                 # Configuration files
├── report/                  # Paper and slides
├── scripts/                 # PowerShell scripts for experiment execution
├── src/
│   ├── data/                # Data preparation and split generation
│   ├── evaluation/          # Evaluation utilities and result aggregation
│   ├── semi_supervised/     # Pseudo-labeling and self-training utilities
│   ├── training/            # DistilHuBERT training code
│   └── utils/               # Shared helper functions
├── README.md
├── requirements.txt
└── .gitignore
```

The following large directories are not included in the GitHub repository:

```text
data/
models/
logs/
results/
```

These directories contain raw datasets, processed datasets, trained checkpoints, logs, generated predictions, evaluation outputs, figures, and tables. They are excluded from GitHub because of file size limitations.

## 4. External Files and Google Drive

Large files are provided separately through Google Drive:

**Google Drive link:**  
`https://drive.google.com/drive/folders/1dYOD1wdzU1XrErahCi9PI4C-1vj9kkUZ?usp=drive_link`

After downloading the Google Drive folder, restore the large directories to the project root using the same directory structure:

```text
data/
models/
logs/
results/
```

### Important Notice About Missing Large Files

Some large files could not be uploaded through Canvas because of size limitations. They are provided in the Google Drive folder instead.

The following trained model directories are especially important and are stored in Google Drive:

```text
models/final/round2_exp5_l2_sg_british_fold1_distilhubert_hfstyle/
models/final/semi_exp7_six_pseudo_conf095_max500_fold0_distilhubert_hfstyle/
models/final/semi_exp7_eight_pseudo_conf095_max500_fold0_distilhubert_hfstyle/
```

The Google Drive folder also contains:

- Raw and processed L2-ARCTIC data
- External accent datasets
- Targeted Singaporean English data
- Processed metadata CSV files
- Speaker-disjoint split CSV files
- Round2 balanced split CSV files
- Semi-supervised pseudo-label CSV files
- Trained model checkpoints
- Evaluation results
- Figures and tables used in the report

## 5. Environment Setup

Create and activate the conda environment:

```bash
conda create -n accent python=3.10 -y
conda activate accent
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project was developed on Windows with an NVIDIA RTX 4060 Laptop GPU. CUDA-enabled PyTorch is recommended.

To verify the environment:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 6. Data Preparation

The main dataset is the official L2-ARCTIC corpus. Processed metadata and split files are expected under:

```text
data/processed/
```

Important processed files include:

```text
data/processed/metadata_l2_arctic_official.csv
data/processed/splits/split2_speaker_disjoint_fold0/
data/processed/splits/split2_speaker_disjoint_fold1/
data/processed/splits/split2_speaker_disjoint_fold2/
data/processed/splits/split2_speaker_disjoint_fold3/
data/processed/round2_balanced/
data/processed/semi_supervised/
data/processed/eval/external_level2_test.csv
```

If the Google Drive data folder is restored, these files should already be available.

## 7. Training

### 7.1 File-Level Baseline

The file-level baseline is used as an upper-bound diagnostic setting. It allows utterances from the same speaker to appear in both training and test sets.

### 7.2 Speaker-Disjoint Baseline

The speaker-disjoint baseline uses four folds. In each fold, one speaker from each L2-ARCTIC accent category is held out for testing.

Example command:

```bash
python -m src.training.train_distilhubert_flexible \
  --model-name ntu-spml/distilhubert \
  --label-mode union \
  --train-csv data/processed/round2_balanced/fold0/train_l2_only.csv \
  --dev-csv data/processed/round2_balanced/fold0/dev_l2_only.csv \
  --test-csv data/processed/round2_balanced/fold0/test_l2_only.csv \
  --output-dir models/final/round2_exp2_l2_only_fold0_distilhubert_hfstyle \
  --epochs 10 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --learning-rate 5e-5 \
  --warmup-ratio 0.1 \
  --max-seconds 8 \
  --freeze-feature-encoder
```

### 7.3 Round2 Balanced External Training

Round2 uses the most promising settings from Round1:

1. L2-only speaker-disjoint training.
2. Expanded eight-class training with L2-ARCTIC plus balanced British Isles English and Singaporean English.

The expanded label set is:

```text
Arabic
Hindi
Korean
Mandarin
Spanish
Vietnamese
OOD_British_Isles_English
OOD_Singaporean_English
```

Round2 uses the following Hugging Face-style configuration:

- Base model: `ntu-spml/distilhubert`
- Learning rate: `5e-5`
- Epochs: `10`
- Scheduler: linear
- Warmup ratio: `0.1`
- Batch size: `4`
- Gradient accumulation steps: `2`
- Effective batch size: `8`
- Maximum audio length: `8` seconds
- Feature encoder: frozen

### 7.4 Semi-Supervised Pseudo-Labeling

Semi-supervised learning is implemented with a teacher-student pseudo-labeling pipeline.

The teacher models are:

1. A six-class L2-only Round2 model.
2. An eight-class expanded Round2 model.

The teacher predicts labels for an external audio pool. Only predictions with confidence at least `0.95` are retained. Each pseudo-label category is capped at `500` samples.

The second-level external evaluation set is excluded from pseudo-label generation to avoid evaluation leakage.

## 8. Evaluation

Evaluation includes:

- Internal L2-ARCTIC accuracy
- Speaker-disjoint test accuracy
- External comparable accuracy
- Expanded-label external accuracy
- Second-level external evaluation accuracy
- Per-label accuracy
- Confusion matrices

First-round result tables are stored under:

```text
results/first_round_all_models_2000/tables/
```

Round2 and semi-supervised outputs are stored under:

```text
results/
models/final/
```

## 9. Important Results

The first-round experiments show:

- File-level evaluation gives inflated accuracy.
- Speaker-disjoint evaluation is more realistic and more difficult.
- External-only training performs poorly on L2-ARCTIC.
- L2-ARCTIC plus external data is more promising than external-only training.
- Expanded-label L2 plus external training is the best Round1 candidate for Round2 balanced training.

The reproduced file-level baseline reaches very high internal accuracy but drops substantially on external data, confirming that high in-corpus accuracy does not guarantee external robustness.

## 10. Notes on Excluded Files

The GitHub repository intentionally excludes large generated artifacts. Do not commit:

```text
data/
models/
logs/
results/
*.pt
*.bin
*.safetensors
*.onnx
*.engine
*.wav
*.flac
*.mp3
*.zip
*.tar
```

Use the Google Drive folder for these files.

## 11. Authorship

All project-specific implementation, experiment organization, data preparation scripts, training scripts, evaluation scripts, and report materials were written and organized by:

He Xu  
UNI: xh2707

## 12. Full Reproduction Pipeline

This section provides the recommended execution order for reproducing the full project pipeline, from baseline verification to external data preparation, Round1 experiments, Round2 balanced training, and semi-supervised learning.

All commands should be executed from the project root directory:

```bash
cd M6600_Project
```

On Windows, activate the conda environment first:

```bash
conda activate accent
```

If using Command Prompt instead of an already activated conda shell:

```cmd
call C:\Users\hexyw\anaconda3\Scripts\activate.bat accent
```

### Step 0: Verify Environment

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Step 1: Run the Public Hugging Face Baseline

This step verifies that the public reference model can be loaded and used for inference.

```bash
python -m src.exp0_run_baseline
```

The reference model is:

```text
kaysrubio/accent-id-distilhubert-finetuned-l2-arctic2
```

### Step 2: Prepare L2-ARCTIC Metadata

Prepare metadata from the Hugging Face L2-ARCTIC version:

```bash
python -m src.data.prepare_l2arctic
```

Prepare metadata from the official L2-ARCTIC release:

```bash
python -m src.data.prepare_l2arctic_official
```

Expected output includes:

```text
data/processed/metadata_l2_arctic.csv
data/processed/metadata_l2_arctic_official.csv
```

### Step 3: Create Speaker-Disjoint Splits

Generate the four speaker-disjoint folds:

```bash
python -m src.data.make_splits
```

Expected output:

```text
data/processed/splits/split2_speaker_disjoint_fold0/
data/processed/splits/split2_speaker_disjoint_fold1/
data/processed/splits/split2_speaker_disjoint_fold2/
data/processed/splits/split2_speaker_disjoint_fold3/
```

Each fold holds out one speaker from each L2-ARCTIC accent category for testing.

### Step 4: Train the File-Level Baseline

The file-level baseline is used as an upper-bound diagnostic setting.

```bash
python -m src.training.run_core_experiments --which file_level --epochs 3 --batch-size 4 --eval-batch-size 4 --gradient-accumulation-steps 2 --max-seconds 8 --freeze-feature-encoder
```

Expected output:

```text
models/checkpoints/exp1_file_level_distilhubert/
```

### Step 5: Train Speaker-Disjoint L2-Only Baselines

Train the speaker-disjoint L2-only models for fold 0 through fold 3:

```bash
python -m src.training.run_core_experiments --which speaker_disjoint --epochs 3 --batch-size 4 --eval-batch-size 4 --gradient-accumulation-steps 2 --max-seconds 8 --freeze-feature-encoder
```

Expected output:

```text
models/checkpoints/exp2_speaker_disjoint_fold0_distilhubert/
models/checkpoints/exp2_speaker_disjoint_fold1_distilhubert/
models/checkpoints/exp2_speaker_disjoint_fold2_distilhubert/
models/checkpoints/exp2_speaker_disjoint_fold3_distilhubert/
```

### Step 6: Prepare General External Accent Data

Prepare the first external metadata pool:

```bash
python -m src.data.prepare_external_datasets --datasets common_accent common_native commonvoice_accent_test english_dialects --max-per-label 1500 --max-total-per-config 2000 --max-scan 200000 --eval
```

Expected output:

```text
data/processed/external_metadata.csv
```

This external pool is intentionally analyzed before balancing because raw external data can be highly imbalanced.

### Step 7: Create First-Round Augmented External Splits

Create augmented splits for Round1 experiments:

```bash
python -m src.data.make_external_augmented_splits
```

Expected output:

```text
data/processed/augmented/fold0/
data/processed/augmented/fold1/
data/processed/augmented/fold2/
data/processed/augmented/fold3/
```

### Step 8: Run Round1 External Experiments

Run Round1 models after external metadata is prepared:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_after_external_ready.ps1
```

This trains the following groups:

```text
Exp3: six-label L2 + external
Exp4: six-label external only
Exp5: expanded-label L2 + external
Exp6: expanded-label external only
```

Expected output:

```text
models/checkpoints/exp3_fold*_six_l2_plus_external_distilhubert/
models/checkpoints/exp4_fold*_six_external_only_distilhubert/
models/checkpoints/exp5_fold*_expanded_l2_plus_external_distilhubert/
models/checkpoints/exp6_fold*_expanded_external_only_distilhubert/
```

### Step 9: Evaluate All Round1 Models

Evaluate the public baseline and all Round1 models on sampled official and external data:

```bash
python -u -m src.exp2_6_first_round_optimization --samples 2000 --seed 42 --output-dir results/first_round_all_models_2000
```

Expected output:

```text
results/first_round_all_models_2000/tables/model_official_external_combined_summary.csv
results/first_round_all_models_2000/tables/model_source_accuracy_summary.csv
results/first_round_all_models_2000/figures/
```

The first-round results are used to select the most promising direction for Round2. In this project, the strongest direction is the expanded L2 plus external setting.

### Step 10: Prepare Targeted Singaporean English Data

Prepare targeted Singaporean English data from MNSC sources:

```bash
python -m src.data.prepare_targeted_external_datasets --datasets mnsc_v1 mnsc_v1_extend --max-per-label 3000 --max-total-per-config 3000 --max-scan 200000
```

Expected output:

```text
data/processed/targeted_external_metadata.csv
```

### Step 11: Merge External and Targeted External Metadata

If the merge is not already included in the targeted preparation script, run:

```bash
python -m src.data.merge_external_metadata
```

Expected output:

```text
data/processed/external_metadata_plus_targeted.csv
```

If this script is not present, the merged metadata file is provided in the Google Drive folder.

### Step 12: Create Round2 Balanced Data

Create balanced Round2 splits with British Isles English and Singaporean English as the two expanded labels:

```bash
python -m src.data.make_round2_balanced_data --new-labels OOD_British_Isles_English OOD_Singaporean_English --target-total-per-new-label 3750 --folds 0 1 2 3 --report-dir results/round2_balanced --output-root data/processed/round2_balanced
```

Expected output:

```text
data/processed/round2_balanced/fold0/
data/processed/round2_balanced/fold1/
data/processed/round2_balanced/fold2/
data/processed/round2_balanced/fold3/
```

Each expanded external label uses approximately:

```text
3000 train samples
375 dev samples
375 test samples
```

### Step 13: Train Round2 Final Supervised Models

Run the visible Round2 final training script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_round2_final_visible.ps1
```

This trains:

```text
Round2 Exp2 L2-only fold0-fold3
Round2 Exp5 L2 + British/Singaporean fold0-fold3
```

Expected output:

```text
models/final/round2_exp2_l2_only_fold0_distilhubert_hfstyle/
models/final/round2_exp2_l2_only_fold1_distilhubert_hfstyle/
models/final/round2_exp2_l2_only_fold2_distilhubert_hfstyle/
models/final/round2_exp2_l2_only_fold3_distilhubert_hfstyle/

models/final/round2_exp5_l2_sg_british_fold0_distilhubert_hfstyle/
models/final/round2_exp5_l2_sg_british_fold1_distilhubert_hfstyle/
models/final/round2_exp5_l2_sg_british_fold2_distilhubert_hfstyle/
models/final/round2_exp5_l2_sg_british_fold3_distilhubert_hfstyle/
```

Round2 uses the Hugging Face-style configuration:

```text
Base model: ntu-spml/distilhubert
Learning rate: 5e-5
Epochs: 10
Warmup ratio: 0.1
Scheduler: linear
Batch size: 4
Gradient accumulation steps: 2
Effective batch size: 8
Maximum audio length: 8 seconds
Feature encoder: frozen
```

### Step 14: Prepare Second-Level External Evaluation Set

Smoke test each source first:

```bash
python -m src.data.prepare_external_level2_eval --datasets esltts globe globe_v2 common_accent --target-labels Arabic Hindi Korean Mandarin Spanish Vietnamese --max-per-label 20 --max-total-per-dataset 300 --max-scan 5000 --max-configs 5 --max-splits 1 --output data/processed/eval/external_level2_smoke_l2.csv
```

```bash
python -m src.data.prepare_external_level2_eval --datasets english_dialects common_accent --target-labels OOD_British_Isles_English --max-per-label 50 --max-total-per-dataset 300 --max-scan 5000 --max-configs 5 --max-splits 1 --output data/processed/eval/external_level2_smoke_british.csv
```

```bash
python -m src.data.prepare_external_level2_eval --datasets mnsc_v1 mnsc_v1_extend --target-labels OOD_Singaporean_English --max-per-label 50 --max-total-per-dataset 300 --max-scan 5000 --max-configs 3 --max-splits 1 --output data/processed/eval/external_level2_smoke_singapore.csv
```

Then prepare the full second-level external evaluation set:

```bash
python -m src.data.prepare_external_level2_eval --datasets esltts globe globe_v2 common_accent english_dialects mnsc_v1 mnsc_v1_extend --target-labels Arabic Hindi Korean Mandarin Spanish Vietnamese OOD_British_Isles_English OOD_Singaporean_English --max-per-label 1000 --max-total-per-dataset 12000 --max-scan 200000 --max-configs 20 --max-splits 2 --output data/processed/eval/external_level2_test.csv
```

Expected output:

```text
data/processed/eval/external_level2_test.csv
```

This file must not be used for supervised training or pseudo-label generation.

### Step 15: Generate Semi-Supervised Pseudo-Labels

First run a smoke test:

```bash
python -m src.semi_supervised.generate_pseudo_labels --teacher-model models/final/round2_exp2_l2_only_fold0_distilhubert_hfstyle/best_model --pool-metadata data/processed/external_metadata_plus_targeted.csv --output-dir data/processed/semi_supervised --name smoke_six --allowed-labels Arabic,Hindi,Korean,Mandarin,Spanish,Vietnamese --exclude-csvs data/processed/round2_balanced/fold0/train_l2_only.csv data/processed/round2_balanced/fold0/dev_l2_only.csv data/processed/round2_balanced/fold0/test_l2_only.csv data/processed/eval/external_level2_test.csv --confidence-threshold 0.90 --max-per-label 20 --max-pool 500 --top-k 6 --seed 42
```

Then run the full semi-supervised pipeline:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_semi_supervised_pipeline.ps1
```

Expected pseudo-label outputs:

```text
data/processed/semi_supervised/six_conf095_max500/
data/processed/semi_supervised/eight_conf095_max500/
data/processed/semi_supervised/fold0_train_semi_six_conf095_max500.csv
data/processed/semi_supervised/fold0_train_semi_eight_conf095_max500.csv
```

Expected student model outputs:

```text
models/final/semi_exp7_six_pseudo_conf095_max500_fold0_distilhubert_hfstyle/
models/final/semi_exp7_eight_pseudo_conf095_max500_fold0_distilhubert_hfstyle/
```

### Step 16: Resume Interrupted Semi-Supervised Training

If training is interrupted and checkpoints exist, resume from the latest checkpoint. Example for the six-class student:

```cmd
call C:\Users\hexyw\anaconda3\Scripts\activate.bat accent && python -u -m src.training.train_distilhubert_flexible --model-name ntu-spml/distilhubert --label-mode union --epochs 10 --batch-size 4 --eval-batch-size 4 --gradient-accumulation-steps 2 --learning-rate 5e-5 --warmup-ratio 0.1 --weight-decay 0.0 --max-seconds 8 --freeze-feature-encoder --logging-steps 1 --seed 42 --train-csv data/processed/semi_supervised/fold0_train_semi_six_conf095_max500.csv --dev-csv data/processed/round2_balanced/fold0/dev_l2_only.csv --test-csv data/processed/round2_balanced/fold0/test_l2_only.csv --output-dir models/final/semi_exp7_six_pseudo_conf095_max500_fold0_distilhubert_hfstyle --resume-from-checkpoint models/final/semi_exp7_six_pseudo_conf095_max500_fold0_distilhubert_hfstyle/checkpoint-7797
```

If an AMP scaler error occurs during resume, rename the scaler file in the checkpoint and retry:

```cmd
ren models\final\semi_exp7_six_pseudo_conf095_max500_fold0_distilhubert_hfstyle\checkpoint-7797\scaler.pt scaler.pt.bak
```

### Step 17: Evaluate Round2 and Semi-Supervised Models

Evaluate Round2 models:

```bash
python -u -m src.exp2_6_first_round_optimization --samples 1000 --seed 42 --only-model round2 --output-dir results/round2_supervised_1000_eval
```

Evaluate semi-supervised models:

```bash
python -u -m src.exp2_6_first_round_optimization --samples 1000 --seed 42 --only-model semi_exp7 --output-dir results/semi_exp7_1000_eval
```

Inspect output tables:

```bash
python -c "import pandas as pd; df=pd.read_csv('results/round2_supervised_1000_eval/tables/model_official_external_combined_summary.csv'); print(df.to_string())"
```

```bash
python -c "import pandas as pd; df=pd.read_csv('results/semi_exp7_1000_eval/tables/model_official_external_combined_summary.csv'); print(df.to_string())"
```

### Step 18: Evaluate on the Second-Level External Set

Use the dedicated evaluation script if available:

```bash
python -m src.evaluation.eval_metadata_models --metadata data/processed/eval/external_level2_test.csv --output-dir results/external_level2_eval --model round2_exp5_fold0 models/final/round2_exp5_l2_sg_british_fold0_distilhubert_hfstyle/best_model --model semi_eight_student models/final/semi_exp7_eight_pseudo_conf095_max500_fold0_distilhubert_hfstyle/best_model
```

Expected output:

```text
results/external_level2_eval/
```

### Step 19: Generate Figures and Tables for the Report

Important generated figures include:

```text
figure/l2arctic_label_distribution.png
figure/l2arctic_label_speaker_heatmap.png
figure/external_label_distribution.png
figure/external_dataset_distribution.png
figure/external_dataset_label_heatmap.png
figure/external_train_after_balancing.png
```

Important result tables include:

```text
results/first_round_all_models_2000/tables/model_official_external_combined_summary.csv
results/first_round_all_models_2000/tables/model_source_accuracy_summary.csv
results/round2_supervised_1000_eval/tables/model_official_external_combined_summary.csv
results/semi_exp7_1000_eval/tables/model_official_external_combined_summary.csv
```

## 13. Recommended Minimal Reproduction

If full reproduction is too expensive, the minimum recommended pipeline is:

```bash
python -m src.exp0_run_baseline
python -m src.data.prepare_l2arctic_official
python -m src.data.make_splits
python -m src.training.run_core_experiments --which file_level --epochs 3 --batch-size 4 --eval-batch-size 4 --gradient-accumulation-steps 2 --max-seconds 8 --freeze-feature-encoder
python -m src.training.run_core_experiments --which speaker_disjoint --epochs 3 --batch-size 4 --eval-batch-size 4 --gradient-accumulation-steps 2 --max-seconds 8 --freeze-feature-encoder
python -u -m src.exp2_6_first_round_optimization --samples 2000 --seed 42 --output-dir results/first_round_all_models_2000
```

This minimal version reproduces the central finding:

```text
File-level evaluation overestimates robustness.
Speaker-disjoint evaluation is more realistic.
External evaluation reveals limited out-of-dataset generalization.
```
