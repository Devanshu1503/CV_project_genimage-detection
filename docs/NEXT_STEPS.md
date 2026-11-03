# Next steps: handoff checklist

Status after phase 1: data pipeline, EDA, preprocessing and Problem 1 baselines are implemented.
Before starting, read `README.md` (how to run), `docs/DATA.md` (data decisions) and `docs/PROJECT_CONTEXT.md` (conventions).

## 0. Finish phase 1 (whoever runs the notebooks)
- [ ] Run notebooks 01 → 02 → 03 on Colab. Fill in every **Findings** cell and the ⚠️ placeholders in `docs/DATA.md`.
- [ ] Share `MyDrive/genimage_project` with all teammates (Editor). Each teammate adds a shortcut to it.
- [ ] Copy key figures/tables into the report draft (EDA section = 15 pts).

## 1. Problem 1 extensions
- [ ] **Grad-CAM on ResNet-50**: which image regions drive "fake"? Compare correct vs wrong predictions
      on seen (SD1.4) and unseen generators. Use `pytorch-grad-cam` (github.com/jacobgil/pytorch-grad-cam)
      or a short custom implementation (bonus points for custom code). Target layer: `model.layer4[-1]`.
      Load `checkpoints/resnet50/best.pt`.
- [ ] **Raw vs processed training comparison**: train the same ResNet-50 on the RAW subset (original
      formats/sizes, random-crop only) and test on raw *and* processed test sets. This quantifies how
      much the format/size shortcut inflates results (expected: great on raw, collapses when fakes are JPEG).
- [ ] Bootstrap 95% confidence intervals for per-generator AUROC (resample test images, 1000×).
- [ ] Optional: look at misclassified images per generator; threshold calibration on SD1.4 val.

## 2. Problem 2: robustness (2 competing models)
- [ ] **Degradation test bench** on the processed test sets: JPEG quality {95, 85, 75, 60, 50} and
      down-/up-scaling {0.5×, 0.75×, 1.5×} (+ optional blur/noise). Report AUROC vs degradation level for
      every model as line charts.
- [ ] **New fakes with SDXL-Turbo** (`diffusers`, `stabilityai/sdxl-turbo`, 1–4 steps, fits a T4 in fp16):
      ~500 images from ImageNet class prompts ("a photo of a {class name}"). Pass them through the
      **same preprocessing** (`image_utils.process_file`) and add them as a new test generator.
      Record prompts, seeds, model version.
- [ ] **Retraining with augmentation**: ResNet-50 (and/or the linear probe on augmented features) trained
      with random JPEG (q 60–100) + random resize + blur, in the spirit of Wang et al. 2020 (arXiv 1912.11035).
      Compare **champion vs challenger**: e.g. ResNet-50 + augmentation vs CLIP probe + augmentation
      (or vs the un-augmented baseline). Same metrics and plots as Problem 1 (`src/evaluate.py`).
- [ ] Add new code to `src/` (e.g. `src/degrade.py`, `src/generate.py`) with tests, and new notebooks
      `04_problem2_robustness.ipynb`, `05_sdxl_turbo_generation.ipynb`. Keep using the shared setup cell.

## 3. MLOps (10 pts)
- [ ] **Deployment architecture** diagram + text: e.g. upload → preprocessing service (the same
      RGB/crop/224/JPEG pipeline, so inputs match training) → model server (TorchServe / FastAPI + ONNX)
      → score + threshold → Grad-CAM explanation on request; logging of inputs/scores; batch vs
      real-time trade-offs; cost/latency of ViT-L vs ResNet-50.
- [ ] **Maintenance & retraining plan**: monitor score distributions and real/fake rates (drift);
      a regular "new generator" test set (like our SDXL-Turbo set) as an early warning; triggers for
      retraining; versioned data (manifests + seeds), models (checkpoints + config hash) and metrics;
      champion/challenger promotion rule (e.g. new model must beat mean-unseen AUROC on a frozen benchmark).
- [ ] Optional demo: small Gradio app loading `best.pt` / the CLIP probe.

## 4. Report (80 pts) and presentation (20 pts)
- [ ] Report sections per rubric: Abstract · EDA · Model training & evaluation (2 problems × 2 models,
      algorithms with pros/cons, detailed metrics) · MLOps · Conclusion + references. Include author
      names and a table of contents.
- [ ] Cite all reused code/ideas (see README references and code comments).
- [ ] Slides: 15 min, ~5 min per member. Suggested split: data and bias (EDA) · Problem 1 · Problem 2 + MLOps.
- [ ] Submit slides + full source code (not the dataset).
