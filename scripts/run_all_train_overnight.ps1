# M6600 Overnight Training Script
# Runs all currently available trainable experiments:
#   Exp1: file-level DistilHuBERT
#   Exp2: speaker-disjoint fold0
#   Exp2: speaker-disjoint fold1
#   Exp2: speaker-disjoint fold2
#   Exp2: speaker-disjoint fold3
#
# Important:
#   - Uses python -u for unbuffered output.
#   - Saves one master log and one detailed log per experiment.
#   - Sets AC sleep / hibernate timeout to Never.
#   - Screen may turn off, but the machine should not sleep.

$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\hexyw\Desktop\M6600_Project"
$CondaRoot = "C:\Users\hexyw\anaconda3"
$EnvName = "accent"

Set-Location $ProjectRoot

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = "logs\overnight_$Timestamp"
New-Item -ItemType Directory -Force $LogDir | Out-Null

$MasterLog = "$LogDir\MASTER_overnight_train.log"

function Write-Log {
    param([string]$Message)

    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $MasterLog -Value $line
}

function Run-Step {
    param(
        [string]$Name,
        [string]$Command,
        [string]$LogFile
    )

    Write-Log "============================================================"
    Write-Log "START: $Name"
    Write-Log "COMMAND: $Command"
    Write-Log "LOG: $LogFile"
    Write-Log "============================================================"

    Add-Content -Path $LogFile -Value "============================================================"
    Add-Content -Path $LogFile -Value "START: $Name"
    Add-Content -Path $LogFile -Value "TIME: $(Get-Date)"
    Add-Content -Path $LogFile -Value "COMMAND: $Command"
    Add-Content -Path $LogFile -Value "============================================================"

    cmd /c "$Command" 2>&1 | Tee-Object -FilePath $LogFile -Append

    if ($LASTEXITCODE -ne 0) {
        Write-Log "FAILED: $Name"
        Write-Log "Exit code: $LASTEXITCODE"
        throw "Step failed: $Name"
    }

    Add-Content -Path $LogFile -Value "============================================================"
    Add-Content -Path $LogFile -Value "END: $Name"
    Add-Content -Path $LogFile -Value "TIME: $(Get-Date)"
    Add-Content -Path $LogFile -Value "============================================================"

    Write-Log "DONE: $Name"
}

Write-Log "M6600 Overnight Training Started"
Write-Log "Project root: $ProjectRoot"
Write-Log "Conda env: $EnvName"
Write-Log "Log dir: $LogDir"

# ---------------------------------------------------------------------
# Prevent sleep / hibernate while training.
# Screen can turn off, but sleep and hibernate should not happen.
# ---------------------------------------------------------------------
Write-Log "Setting AC screen timeout to 10 minutes."
Write-Log "Setting AC sleep and hibernate timeout to Never."

powercfg /change monitor-timeout-ac 10
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

# Conda activation command used inside cmd /c.
$Activate = "call `"$CondaRoot\Scripts\activate.bat`" $EnvName"

# ---------------------------------------------------------------------
# 00. Environment check
# ---------------------------------------------------------------------
Run-Step `
    -Name "00 Check CUDA and imports" `
    -Command "$Activate && python -u -c `"import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); from src.utils.constants import L2ARCTIC_OFFICIAL_METADATA; print('metadata path:', L2ARCTIC_OFFICIAL_METADATA)`"" `
    -LogFile "$LogDir\00_check_env.log"

# ---------------------------------------------------------------------
# 01. Make splits
# ---------------------------------------------------------------------
Run-Step `
    -Name "01 Make data splits" `
    -Command "$Activate && python -u -m src.data.make_splits" `
    -LogFile "$LogDir\01_make_splits.log"

# ---------------------------------------------------------------------
# 02. Verify speaker-disjoint split
# ---------------------------------------------------------------------
Run-Step `
    -Name "02 Verify speaker-disjoint split" `
    -Command "$Activate && python -u -c `"import pandas as pd; tr=pd.read_csv('data/processed/train_speaker_disjoint.csv'); te=pd.read_csv('data/processed/test_speaker_disjoint.csv'); ov=set(tr['speaker']) & set(te['speaker']); print('speaker overlap =', ov); assert len(ov)==0`"" `
    -LogFile "$LogDir\02_verify_speaker_disjoint.log"

# ---------------------------------------------------------------------
# Main training settings.
#
# If CUDA OOM happens, change:
#   --batch-size 4 -> --batch-size 2
#   --eval-batch-size 4 -> --eval-batch-size 2
#   --gradient-accumulation-steps 2 -> --gradient-accumulation-steps 4
# ---------------------------------------------------------------------
$CommonArgs = "--epochs 3 --batch-size 4 --eval-batch-size 4 --gradient-accumulation-steps 2 --max-seconds 8 --freeze-feature-encoder --logging-steps 10"

# ---------------------------------------------------------------------
# 03. Exp1 file-level
# ---------------------------------------------------------------------
Run-Step `
    -Name "03 Exp1 file-level DistilHuBERT" `
    -Command "$Activate && python -u -m src.training.train_distilhubert --train-csv data/processed/train_file_level.csv --dev-csv data/processed/val_file_level.csv --test-csv data/processed/test_file_level.csv --output-dir models/checkpoints/exp1_file_level_distilhubert $CommonArgs" `
    -LogFile "$LogDir\03_exp1_file_level.log"

# ---------------------------------------------------------------------
# 04. Exp2 speaker-disjoint fold0
# ---------------------------------------------------------------------
Run-Step `
    -Name "04 Exp2 speaker-disjoint fold0 DistilHuBERT" `
    -Command "$Activate && python -u -m src.training.train_distilhubert --train-csv data/processed/splits/split2_speaker_disjoint_fold0/train.csv --dev-csv data/processed/splits/split2_speaker_disjoint_fold0/dev.csv --test-csv data/processed/splits/split2_speaker_disjoint_fold0/test.csv --output-dir models/checkpoints/exp2_speaker_disjoint_fold0_distilhubert $CommonArgs" `
    -LogFile "$LogDir\04_exp2_fold0.log"

# ---------------------------------------------------------------------
# 05. Exp2 speaker-disjoint fold1
# ---------------------------------------------------------------------
Run-Step `
    -Name "05 Exp2 speaker-disjoint fold1 DistilHuBERT" `
    -Command "$Activate && python -u -m src.training.train_distilhubert --train-csv data/processed/splits/split2_speaker_disjoint_fold1/train.csv --dev-csv data/processed/splits/split2_speaker_disjoint_fold1/dev.csv --test-csv data/processed/splits/split2_speaker_disjoint_fold1/test.csv --output-dir models/checkpoints/exp2_speaker_disjoint_fold1_distilhubert $CommonArgs" `
    -LogFile "$LogDir\05_exp2_fold1.log"

# ---------------------------------------------------------------------
# 06. Exp2 speaker-disjoint fold2
# ---------------------------------------------------------------------
Run-Step `
    -Name "06 Exp2 speaker-disjoint fold2 DistilHuBERT" `
    -Command "$Activate && python -u -m src.training.train_distilhubert --train-csv data/processed/splits/split2_speaker_disjoint_fold2/train.csv --dev-csv data/processed/splits/split2_speaker_disjoint_fold2/dev.csv --test-csv data/processed/splits/split2_speaker_disjoint_fold2/test.csv --output-dir models/checkpoints/exp2_speaker_disjoint_fold2_distilhubert $CommonArgs" `
    -LogFile "$LogDir\06_exp2_fold2.log"

# ---------------------------------------------------------------------
# 07. Exp2 speaker-disjoint fold3
# ---------------------------------------------------------------------
Run-Step `
    -Name "07 Exp2 speaker-disjoint fold3 DistilHuBERT" `
    -Command "$Activate && python -u -m src.training.train_distilhubert --train-csv data/processed/splits/split2_speaker_disjoint_fold3/train.csv --dev-csv data/processed/splits/split2_speaker_disjoint_fold3/dev.csv --test-csv data/processed/splits/split2_speaker_disjoint_fold3/test.csv --output-dir models/checkpoints/exp2_speaker_disjoint_fold3_distilhubert $CommonArgs" `
    -LogFile "$LogDir\07_exp2_fold3.log"

# ---------------------------------------------------------------------
# 08. Collect summaries
# ---------------------------------------------------------------------
Run-Step `
    -Name "08 Collect training summaries" `
    -Command "$Activate && python -u -c `"from pathlib import Path; paths=sorted(Path('models/checkpoints').glob('exp*_distilhubert/test_summary.json')); print('Found summaries:', len(paths)); [print('\n'+str(p)+'\n'+p.read_text(encoding='utf-8')) for p in paths]`"" `
    -LogFile "$LogDir\08_collect_summaries.log"

Write-Log "============================================================"
Write-Log "ALL TRAINING FINISHED SUCCESSFULLY"
Write-Log "End time: $(Get-Date)"
Write-Log "Logs saved to: $LogDir"
Write-Log "============================================================"