# Do Non-Ocular Features Add Predictive Value Beyond Ocular Symptoms?
### A Feature-Group Ablation and Explanation-Stability Analysis of Dry Eye Disease Classification

Code, notebooks and result files for the paper of the same name.

## What is here

| Path | Content |
|---|---|
| `notebooks/01_main_pipeline.ipynb` | Main pipeline: data audit, five feature groups, locked 80/20 split, baselines, ablation, incremental groups, repeated CV with Wilcoxon/Holm, SHAP and SHAP stability, calibration, held-out test |
| `notebooks/02_extra_checks.ipynb` | Extra checks: label-permutation test, ocular pattern table, split sensitivity, add-one-feature test, screen-time re-run, paper figures |
| `scripts/` | The same extra checks as command-line scripts |
| `results/` | Result files quoted in the paper |
| `figures/` | Paper figures (PDF) |
| `data/` | Put the Kaggle CSV here (not included, see `data/README.md`) |

## How to run

```bash
pip install -r requirements.txt
# 1. download the CSV from Kaggle into data/
# 2. run the notebooks in order
jupyter nbconvert --to notebook --execute notebooks/01_main_pipeline.ipynb --output-dir outputs
jupyter nbconvert --to notebook --execute notebooks/02_extra_checks.ipynb --output-dir outputs
```


## Reproducibility notes

* Seed 42 is used everywhere. The split is stratified 80/20.
* XGBoost results can move by about 0.001-0.002 ROC-AUC between library versions. The paper numbers come from a Kaggle notebook. A re-run with a newer XGBoost gave 0.6006 instead of 0.6016 (all features) and 0.5026 instead of 0.5045 (no ocular) on the test set. The conclusions do not change.
* Logistic Regression results reproduce exactly.
* The smoke test of this release used a synthetic stand-in file with the same columns, only to check that the code runs. It was not used for any number in the paper.
* The notebook outputs are stripped to keep the files small. Run them to regenerate tables and figures.

## Data

Kaggle Dry Eye Disease data set: https://www.kaggle.com/datasets/dakshnagra/dry-eye-disease
The origin of the data is not documented. See the Limitations section of the paper.

## Citation
.....
