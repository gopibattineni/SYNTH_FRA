@echo off
REM Windows helper: show commands to run on the Hive login node.
REM This file does NOT submit to SLURM from Windows — copy the project to Hive first.

setlocal
set "HIVE_DIR=%~dp0"
set "PROJECT_ROOT=%HIVE_DIR%..\.."

echo ============================================================
echo FORGE-PAPER - Hive parallel dataset launcher
echo ============================================================
echo.
echo Project root (local):
echo   %PROJECT_ROOT%
echo.
echo CLI arguments supported by cancer.py:
echo   --uci-id        UCI ML dataset ID
echo   --n-samples     Synthetic rows per generator
echo   --test-size     Test split fraction (default 0.2)
echo   --seed          Random seed (default 42)
echo   --train-csv     CTAB-GAN training CSV filename
echo   --output-file   Excel results filename
echo   --notebook-dir  Dataset working directory
echo.
echo Datasets (see datasets.json):
echo   0  cancer  - UCI 17  - 1. Cancer
echo   1  magic   - UCI 159 - 2. MAGIC Gamma Telescope
echo   2  adult   - UCI 2   - 3. Adult          (nbconvert)
echo   3  bank    - UCI 222 - 5. Bank Markting   (nbconvert)
echo   4  wine    - UCI 186 - 6. Wine dataset    (nbconvert)
echo.
echo On Hive login node, run:
echo   cd python_scripts/hive
echo   mkdir logs
echo   export FORGE_PROJECT_ROOT=/path/to/Single run_Data_leak_Synth_Quality
echo   export FORGE_CONDA_ENV=your-conda-env
echo   sbatch submit_parallel.slurm
echo.
echo Run one dataset manually on Hive:
echo   bash run_one_dataset.sh cancer
echo   bash run_one_dataset.sh 2
echo.
echo Example single cancer.py command:
echo   python "../1. Cancer/cancer.py" ^
echo     --uci-id 17 ^
echo     --n-samples 1000 ^
echo     --notebook-dir "../../1. Cancer" ^
echo     --train-csv breast_cancer_train.csv ^
echo     --output-file TRTR_TSTR_results_cancer.xlsx
echo.
pause
