# docs/PROJECT_CONTEXT.md: project context for AI coding assistants

## Project
UChicago MSADS Advanced Computer Vision team project: **detecting AI-generated images** on GenImage
(real ImageNet vs fakes from 8 generators). Rubric: EDA, 2 problems × 2 competing models with
detailed metrics, MLOps (deployment + maintenance), report, presentation.
- Problem 1 (done in this phase): train on SD1.4 only, test on all 8 generators. Fine-tuned ResNet-50 vs
  frozen CLIP ViT-L/14 / DINOv2 ViT-B/14 + logistic regression. Main metric: AUROC per generator.
- Problem 2 (next): robustness to JPEG/resizing, new SDXL-Turbo fakes, augmentation retraining.
- Handoff checklist: docs/NEXT_STEPS.md. Dataset facts and preprocessing decisions: docs/DATA.md.

## Environment
- Repo: https://github.com/Devanshu1503/CV_project_genimage-detection (branch `main`); Colab clones it to
  `/content/CV_project_genimage-detection`.
- Dependencies: `requirements.txt` (local) / `requirements-colab.txt` (Colab, installed only if missing:
  never reinstall/upgrade Colab's torch, numpy, pandas, etc.).
- Code is written locally (Windows, CPU), **run on Google Colab**. Data lives on Google Drive:
  - `/content/drive/MyDrive/genimage/<generator>/`: read-only split zips (`<base>.z01..zNN` + `<base>.zip`)
  - `/content/drive/MyDrive/genimage_project/`: our outputs (manifests, raw_subset, processed, features,
    checkpoints, results, figures, logs). Keep it small (Drive budget in config).
- Never copy or extract a whole generator archive. Read archives in place with `7z l -slt` / `7z x -i@list`.

## Conventions
- **All paths, seeds, sizes and hyperparameters come from `configs/config.yaml`** via `src/config.py`.
  Never hardcode them.
- Notebooks are thin: logic goes in `src/`, notebooks call it and show outputs. Every notebook starts
  with the identical setup cell and ends with a "Findings" markdown cell.
- Everything must be **resumable** (Colab disconnects): skip finished work, checkpoint long loops.
- Labels: `y = 1` = fake (`ai`), `y = 0` = real (`nature`). Scores are P(fake).
- CSV paths are **relative** to `genimage_project/` (`raw_relpath`, `processed_relpath`).
  Read project CSVs with `src.manifest.read_csv` (keeps md5/phash/class ids as strings).
- Single source of truth: `manifests/subset_index.csv` (raw) → `processed_index.csv` + `splits.csv`.
- Every image, real and fake, goes through the same preprocessing (`image_utils.preprocess_image` + `save_jpeg`).
- Beginner-friendly comments (what + why). Clear printed messages; never silently swallow errors.
- Cite sources (papers / repos) in comments and docs when reusing ideas.
- Don't invent dataset facts. If something must be verified, write code that checks and reports it.
- Tests: `python -m pytest -q` must pass locally without data. Add tests for new pure functions.
