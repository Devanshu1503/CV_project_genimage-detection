# Next steps

Where we are: notebooks 01 (dataset) and 02 (EDA + preprocessing) have been run, see `docs/FINDINGS.md`.
Notebook 03 (Problem 1 baselines) is next. Before starting, read `README.md` (how to run), `docs/DATA.md`
(data decisions) and `docs/PROJECT_CONTEXT.md` (code conventions).

## 0. Finish phase 1
- [x] Run notebooks 01 and 02 on Colab, fill in their Findings cells and `docs/DATA.md`.
- [x] Run notebook 03 and fill in its Findings cell.
- [ ] Share `MyDrive/genimage_project` with all teammates (Editor). Each teammate adds a shortcut to it.
- [ ] Copy the key figures and tables into the report draft (the EDA section is worth 15 pts).

## 1. Problem 1 extensions
[x] = code written and tested locally (notebooks 04, 06, 07, 08). Still needs a Colab run + Findings cells filled in.

- [x] Grad-CAM on ResNet-50: which image regions push it towards "fake"? Compare correct vs wrong predictions
      on seen (SD1.4) and unseen generators. Use `pytorch-grad-cam` (github.com/jacobgil/pytorch-grad-cam)
      or write a short version ourselves (bonus points for own code). Target layer: `model.layer4[-1]`.
      Load `checkpoints/resnet50/best.pt`.
- [ ] Raw vs processed training: train the same ResNet-50 on the raw subset (original formats and sizes,
      random crop only) and test on both raw and processed test sets. This shows how much the format/size
      shortcut inflates results. We expect it to look great on raw and fall apart once fakes are JPEG.
- [x] Bootstrap 95% confidence intervals for per-generator AUROC (resample test images 1000 times).
- [ ] Optional: look at misclassified images per generator, calibrate the threshold on SD1.4 val.

## 2. Problem 2: robustness (2 competing models)
- [x] Degradation tests on the processed test sets: JPEG quality {95, 85, 75, 60, 50} and
      down/up-scaling {0.5x, 0.75x, 1.5x} (blur and noise optional). Plot AUROC vs degradation level for
      every model as line charts.
- [x] New fakes with SDXL-Turbo (`diffusers`, `stabilityai/sdxl-turbo`, 1-4 steps, fits on a T4 in fp16):
      ~500 images from ImageNet class prompts ("a photo of a {class name}"). Run them through the
      same preprocessing (`image_utils.process_file`) and add them as a new test generator.
      Write down prompts, seeds and model version.
- [x] Retraining with augmentation: ResNet-50 (and/or the linear probe on augmented features) trained
      with random JPEG (q 60-100), random resize and blur, like Wang et al. 2020 (arXiv 1912.11035).
      Compare champion vs challenger, e.g. ResNet-50 + augmentation vs CLIP probe + augmentation
      (or vs the baseline without augmentation). Same metrics and plots as Problem 1 (`src/evaluate.py`).
- [x] Put new code in `src/` (e.g. `src/degrade.py`, `src/generate.py`) with tests, and add notebooks
      `04_problem2_robustness.ipynb` and `05_sdxl_turbo_generation.ipynb`. Keep using the shared setup cell.

## 3. MLOps (10 pts)
- [ ] Deployment diagram + short text, e.g. upload -> preprocessing (the same RGB/crop/224/JPEG steps,
      so inputs look like training data) -> model server (TorchServe or FastAPI + ONNX)
      -> score + threshold -> Grad-CAM on request. Also cover logging inputs and scores, batch vs
      real-time, and cost/latency of ViT-L vs ResNet-50.
- [ ] Maintenance and retraining plan: watch score distributions and real/fake rates for drift,
      keep a regular "new generator" test set (like our SDXL-Turbo set) as an early warning, decide when to
      retrain, version data (manifests + seeds), models (checkpoints + config hash) and metrics,
      and set a rule for swapping in a new model (e.g. it has to beat mean unseen AUROC on a fixed benchmark).
- [ ] Optional demo: small Gradio app that loads `best.pt` or the CLIP probe.

## 4. Report (80 pts) and presentation (20 pts)
- [ ] Report sections per rubric: abstract, EDA, model training and evaluation (2 problems x 2 models,
      pros/cons of each, detailed metrics), MLOps, conclusion + references. Add author names and a
      table of contents.
- [ ] Cite all reused code and ideas (see the README references and code comments).
- [ ] Slides: 15 min, about 5 min each. Possible split: data and bias (EDA), Problem 1, Problem 2 + MLOps.
- [ ] Submit slides + full source code (not the dataset).
