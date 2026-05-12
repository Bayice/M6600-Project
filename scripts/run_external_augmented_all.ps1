$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\hexyw\Desktop\M6600_Project"
$CondaRoot = "C:\Users\hexyw\anaconda3"
$EnvName = "accent"

Set-Location $ProjectRoot

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = "logs\external_augmented_$Timestamp"
New-Item -ItemType Directory -Force $LogDir | Out-Null

$MasterLog = "$LogDir\MASTER_external_augmented.log"

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

Write-Log "External augmented experiment started"
Write-Log "Project root: $ProjectRoot"
Write-Log "Log dir: $LogDir"

powercfg /change monitor-timeout-ac 10
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

$Activate = "call `"$CondaRoot\Scripts\activate.bat`" $EnvName"

Run-Step `
    -Name "00 Check CUDA and import" `
    -Command "$Activate && python -u -c `"import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')`"" `
    -LogFile "$LogDir\00_check_env.log"

Run-Step `
    -Name "01 Prepare external datasets" `
    -Command "$Activate && python -u -m src.data.prepare_external_datasets --datasets common_accent common_native commonvoice_accent_test english_dialects --max-per-label 1000 --max-total-per-config 2000 --max-scan 100000 --eval" `
    -LogFile "$LogDir\01_prepare_external_datasets.log"

Run-Step `
    -Name "02 Make external augmented train CSVs" `
    -Command "$Activate && python -u -m src.data.make_external_augmented_splits" `
    -LogFile "$LogDir\02_make_external_augmented_splits.log"

$CommonTrainArgs = "--epochs 3 --batch-size 4 --eval-batch-size 4 --gradient-accumulation-steps 2 --max-seconds 8 --freeze-feature-encoder --logging-steps 10"

for ($Fold = 0; $Fold -lt 4; $Fold++) {
    $FoldDir = "data/processed/augmented/fold$Fold"
    $DevCsv = "$FoldDir/dev.csv"
    $TestCsv = "$FoldDir/test.csv"

    Run-Step `
        -Name "fold$Fold A1 six-label L2 plus external" `
        -Command "$Activate && python -u -m src.training.train_distilhubert_flexible --label-mode baseline --train-csv $FoldDir/train_six_l2_plus_external.csv --dev-csv $DevCsv --test-csv $TestCsv --output-dir models/checkpoints/exp3_fold${Fold}_six_l2_plus_external_distilhubert $CommonTrainArgs" `
        -LogFile "$LogDir\03_fold${Fold}_six_l2_plus_external.log"

    Run-Step `
        -Name "fold$Fold A2 six-label external only" `
        -Command "$Activate && python -u -m src.training.train_distilhubert_flexible --label-mode baseline --train-csv $FoldDir/train_six_external_only.csv --dev-csv $DevCsv --test-csv $TestCsv --output-dir models/checkpoints/exp4_fold${Fold}_six_external_only_distilhubert $CommonTrainArgs" `
        -LogFile "$LogDir\04_fold${Fold}_six_external_only.log"

    Run-Step `
        -Name "fold$Fold B1 expanded-label L2 plus external" `
        -Command "$Activate && python -u -m src.training.train_distilhubert_flexible --label-mode union --train-csv $FoldDir/train_expanded_l2_plus_external.csv --dev-csv $DevCsv --test-csv $TestCsv --output-dir models/checkpoints/exp5_fold${Fold}_expanded_l2_plus_external_distilhubert $CommonTrainArgs" `
        -LogFile "$LogDir\05_fold${Fold}_expanded_l2_plus_external.log"

    Run-Step `
        -Name "fold$Fold B2 expanded-label external only" `
        -Command "$Activate && python -u -m src.training.train_distilhubert_flexible --label-mode union --train-csv $FoldDir/train_expanded_external_only.csv --dev-csv $DevCsv --test-csv $TestCsv --output-dir models/checkpoints/exp6_fold${Fold}_expanded_external_only_distilhubert $CommonTrainArgs" `
        -LogFile "$LogDir\06_fold${Fold}_expanded_external_only.log"
}

Run-Step `
    -Name "Collect summaries" `
    -Command "$Activate && python -u -c `"from pathlib import Path; paths=sorted(Path('models/checkpoints').glob('exp*_distilhubert/test_summary.json')); print('Found summaries:', len(paths)); [print('\n'+str(p)+'\n'+p.read_text(encoding='utf-8')) for p in paths]`"" `
    -LogFile "$LogDir\99_collect_summaries.log"

Write-Log "============================================================"
Write-Log "ALL EXTERNAL AUGMENTED EXPERIMENTS FINISHED"
Write-Log "Logs saved to: $LogDir"
Write-Log "============================================================"