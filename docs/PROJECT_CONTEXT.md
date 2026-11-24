# Project context: GenImage fake image detection (team project)

Context file for anyone on the team working on this repo. If you use an AI assistant, give it this file first.
Then read README.md and docs/FINDINGS.md.

## The project in 30 seconds

- Class: UChicago MSADS, Advanced Computer Vision (ADSP 31023), final team project.
- Team: Mahima Goyal, Devanshu Khadka, Sadaf Khan, Sameeksha Kini, Anuradha Rakh.
- Goal: build fake image detectors (real photo vs AI-generated) and test if they still work on image generators they never saw during training.
- Data: GenImage (Zhu et al. 2023, arXiv 2306.08571). Real ImageNet photos plus fakes from 8 generators: SD1.4, SD1.5, Wukong, Midjourney, ADM, GLIDE, VQDM (diffusion models) and BigGAN (a GAN). The fakes come pre-made in the dataset, we did not generate them.
- Repo: https://github.com/Devanshu1503/CV_project_genimage-detection
- Code is written locally, pushed to GitHub, and run on Google Colab. Data lives on Google Drive.

## Rubric (what we get graded on)

- Report (80): abstract 5, EDA 15, model training + evaluation 40, MLOps 10, conclusion 10.
- Model part needs 2 different problems, each with at least 2 competing models (champion vs challenger), with detailed metrics (accuracy, F1, ROC/AUROC, confusion matrices) and pros/cons of each algorithm. Bonus for custom code.
- MLOps: deployment architecture + maintenance / retraining plan.
- Presentation (20): 15 minutes total, everyone presents their part.
- Submit slides + full source code, not the dataset. Cite all reused code and papers. Report needs author names and a table of contents.

## What is done

### Notebook 01: build the dataset (done)
- GenImage is 656 GB of split zip files on a shared Drive folder. We read the zips in place with 7z and only extracted a sample.
- Sample: 18,000 images, 4.91 GB.
  - Training pool: 5000 real + 5000 SD1.4 fakes.
  - Test sets: 500 real + 500 fake for each of the 8 generators.
- 0 corrupt files, 0 dropped, 0 train/test leakage (checked by filename, md5 and perceptual hash).
- 2 near-duplicate pairs between SD1.4 and SD1.5 test fakes (same prompt + seed, both test, kept).

### Notebook 02: EDA + preprocessing (done)
- Big bias in GenImage: every real image is a JPEG (median quality 96), every fake is a PNG at one fixed size per generator (SD/Wukong 512, Midjourney 1024, ADM/GLIDE/VQDM 256, BigGAN 128).
- A decision tree that only sees file metadata (no pixels) learned "is_jpeg" and got 100% accuracy. So a detector could cheat.
- Fix: every image (real and fake) is center cropped, resized to 224x224 and saved as JPEG quality 96. After that the same metadata test drops to 50% (chance).
- Other EDA: brightness KS D = 0.079, BigGAN shows a grid of FFT peaks (GAN upsampling artifacts), SD1.4/SD1.5/Wukong have extra high-frequency energy that survives preprocessing, outliers kept (149 tiny reals, 170 grayscale reals, 17 grayscale Midjourney).
- Splits: train 4000 + 4000 (SD1.4), val 1000 + 1000 (SD1.4), test 500 + 500 per generator.

### Notebook 03: Problem 1 baselines (done)
Problem 1 = cross-generator detection. Train on real vs SD1.4 fakes only, test on all 8 generators.

Three detectors:
- ResNet-50, fine-tuned (champion): ImageNet-pretrained, last layer replaced by 1 output, all weights trained for 5 epochs, AdamW, lr 1e-4 cosine, batch 64, random 192 px crop + flip (no resize), AMP. ~35 s/epoch on a T4.
- CLIP ViT-L/14 + logistic regression (challenger): CLIP is frozen (weights not changed). We extract a 768-number feature vector per image and train only a logistic regression on top. C tuned on val, best C = 0.01.
- DINOv2 ViT-B/14 + logistic regression (challenger): same idea as CLIP. Best C = 0.01.

AUROC on the test sets:

| Model | SD1.4 (seen) | SD1.5 | Wukong | Midjourney | GLIDE | VQDM | ADM | BigGAN | Mean unseen |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | 0.991 | 0.995 | 0.989 | 0.833 | 0.615 | 0.613 | 0.624 | 0.424 | 0.728 |
| CLIP + LR | 0.985 | 0.987 | 0.946 | 0.930 | 0.871 | 0.735 | 0.756 | 0.646 | 0.839 |
| DINOv2 + LR | 0.935 | 0.937 | 0.914 | 0.870 | 0.721 | 0.584 | 0.526 | 0.397 | 0.707 |

Takeaways:
- ResNet-50 is best on what it was trained on, CLIP generalizes best to unseen generators.
- SD1.5 and Wukong are easy for everyone (same latent diffusion design as SD1.4).
- On unseen generators, real-image accuracy stays at 0.93-0.96. The errors are new fakes being called real (ResNet fake accuracy: ADM 0.07, GLIDE 0.10, VQDM 0.11, BigGAN 0.01).
- BigGAN AUROC is below 0.5 for ResNet and DINOv2, likely because 128 px BigGAN images get upscaled (blurred) and lack the high-frequency traces learned from SD.

## Notebooks 04, 06, 07, 08 (written, not yet run on Colab)

- 04 Grad-CAM + bootstrap: own Grad-CAM (`src/gradcam.py`, checked against a manual computation in tests), gallery of caught/missed fakes on seen vs unseen generators, mean heatmaps, deletion test. Bootstrap CIs + paired bootstrap model differences (`src/stats.py`). Outputs: `results/problem1_extras/`, `figures/problem1_extras/`.
- 06 degradation: 14 conditions in `config.yaml > problem2.conditions` (JPEG 95-50, resize 0.5/0.75/1.5 round trip, blur, noise, resize+JPEG "social"), applied to real and fake. Scores all 3 Problem 1 models. Clean condition is checked against Problem 1 numbers.
- 07 SDXL-Turbo: 500 fakes, prompts/seeds in `problem2/sdxl_turbo/prompts.csv`, versions in `generation_meta.json`, same preprocessing, test index `problem2/sdxl_turbo/sdxl_index.csv` (reals = SD1.4 test reals).
- 08 augmentation: ResNet-50 + random blur/resize/JPEG (champion) vs CLIP probe on clean + 2 augmented feature copies (challenger), plus both baselines, on GenImage + SDXL-Turbo under all conditions. Bootstrap CIs + paired tests on clean and "social".
- Shared code: `src/degrade.py`, `src/robustness.py` (cached per model x test set x condition in `results/problem2/preds/`, CLIP features in `features/problem2/`), `src/generate.py`.
- Order: 06 and 07 before 08.

## What is left

Pick one workstream each. Suggested notebook names are there so we don't edit the same files.

1. Problem 1 extras (`04_problem1_gradcam.ipynb`)
   - Grad-CAM heatmaps on ResNet-50 (`checkpoints/resnet50/best.pt`, target layer `model.layer4[-1]`). Compare correct vs wrong predictions on seen and unseen generators. Writing our own Grad-CAM gets bonus points, or use pytorch-grad-cam and cite it.
   - Bootstrap 95% confidence intervals for per-generator AUROC (resample test images 1000 times) using `results/problem1_predictions.csv`.

2. Raw vs processed (`05_raw_vs_processed.ipynb`)
   - Train the same ResNet-50 on the RAW subset (`raw_subset/`, original JPEG/PNG and sizes) and test on raw and processed test sets.
   - Shows how much the JPEG/PNG shortcut inflates results. Expected: near perfect on raw, collapses when fakes are JPEG.

3. Problem 2a: degradation tests (`06_problem2_degradation.ipynb`)
   - Apply JPEG quality {95, 85, 75, 60, 50} and resize {0.5x, 0.75x, 1.5x} (optional blur/noise) to the processed test sets.
   - Score all 3 Problem 1 detectors (saved checkpoints, no retraining), plot AUROC vs degradation level.

4. Problem 2b: new generator + retraining (`07_sdxl_turbo_generation.ipynb`, `08_problem2_augmentation.ipynb`)
   - Generate ~500 fakes with SDXL-Turbo (`stabilityai/sdxl-turbo`, diffusers, 1-4 steps, fp16 on a T4) from ImageNet class prompts ("a photo of a {class name}"). Run them through the SAME preprocessing (`image_utils.process_file`). Save prompts, seeds, model version.
   - Retrain with augmentation (random JPEG q 60-100, random resize, blur), like Wang et al. 2020 (arXiv 1912.11035).
   - 2 competing models, e.g. ResNet-50 + augmentation vs CLIP probe + augmentation. Same metrics/plots as Problem 1 (`src/evaluate.py`).

5. MLOps (`docs/MLOPS.md`, optional demo)
   - Deployment diagram: upload, same preprocessing (crop, 224, JPEG q96), model server (FastAPI or TorchServe, maybe ONNX), score + threshold, Grad-CAM on request, logging.
   - Maintenance plan: drift monitoring on score distributions, a "new generator" test set as an early warning, retraining triggers, versioning (data manifests + seeds, checkpoints + config), champion/challenger promotion rule.
   - Optional: small Gradio demo using `best.pt` or the CLIP probe.

6. Report + slides: everyone writes their own section. 15 min presentation, everyone presents their part.

Open questions for the professor:
- Does robustness count as a separate second problem? If not, backup Problem 2 = generator attribution (predict which of the 8 generators made an image, same data, 8 classes).
- We are 5 people and the rubric says 3-4.

## Setup (one time)

1. Get access: Devanshu shares the Drive folder `genimage_project` with you (Editor) and adds you to the GitHub repo.
2. In Google Drive, go to Shared with me, right click `genimage_project` > Organize > Add shortcut > My Drive. Keep the name exactly `genimage_project`.
3. Only needed if you want to rerun notebook 01 (you don't): add a shortcut to the GenImage folder named `genimage` (https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS).
4. Clone the repo locally: `git clone https://github.com/Devanshu1503/CV_project_genimage-detection.git`
5. Local tests (no data needed): `pip install -r requirements.txt` then `python -m pytest -q`.
6. To run a notebook: click its "Open in Colab" badge in the README. The first cell clones/pulls the repo, mounts Drive, installs missing packages and sets seeds. Use the same Google account that has the `genimage_project` shortcut.
7. Runtime: CPU for data/EDA work, T4 GPU for anything that trains or extracts features (Runtime > Change runtime type).

## Where things are

```
configs/config.yaml    all paths, sample sizes, preprocessing and training settings
src/                   the real code, notebooks just call it
  archive_utils.py     7z listing/extraction
  manifest.py          manifests, sampling, leakage/duplicate checks
  image_utils.py       metadata, JPEG quality, hashing, preprocessing (process_file), FFT
  eda_utils.py         plots, pixel stats, shortcut test
  datasets.py          PyTorch Dataset + transforms, load_index()
  models.py            ResNet-50, CLIP/DINOv2 feature extractors, feature caching
  train.py             ResNet training loop, linear probe
  evaluate.py          metrics, per-generator table, ROC/confusion/bar plots
  colab_setup.py       setup() and flush_drive()
notebooks/             clean notebooks (no outputs)
executed_notebooks/    Colab runs with outputs, record only
tests/                 pytest, runs locally without data
docs/                  DATA.md, FINDINGS.md, NEXT_STEPS.md
```

On Drive, `MyDrive/genimage_project/`:
- `manifests/processed_index.csv` + `splits.csv`: the main index for any modeling (use `datasets.load_index`).
- `processed/`: the 18,000 preprocessed 224 px JPEGs (use these, not raw, unless your task is raw vs processed).
- `raw_subset/`: the original files (only for raw vs processed).
- `features/clip`, `features/dinov2`: cached embeddings (.npy).
- `checkpoints/resnet50/best.pt`, `checkpoints/linear_probe_clip.joblib`, `checkpoints/linear_probe_dinov2.joblib`: trained Problem 1 models.
- `results/problem1_predictions.csv`, `results/problem1_results.csv`: scores and metrics.
- `figures/`, `logs/`.

## Rules for working on shared stuff

Drive (we learned these the hard way):
- Never use "Disconnect and delete runtime" while files are being written to Drive. Colab uploads in the background and unflushed files get lost.
- If something hangs, use Runtime > Interrupt execution.
- Run the last cell ("Run before closing the tab", `flush_drive()`) before closing a tab.
- Only one Colab session writing to Drive at a time per person.
- Do NOT rerun notebooks 01 or 02 and do NOT write into `raw_subset/`, `processed/` or `manifests/`. Everyone depends on those.
- Write your outputs to your own subfolders, e.g. `results/<your_workstream>/`, `figures/<your_workstream>/`, `checkpoints/<your_workstream>/`.

Git:
- `git pull` before you start, every time.
- Work on your own branch (`git checkout -b <yourname>-<task>`) and open a pull request, or at least only touch your own notebook + new src files.
- Don't edit someone else's notebook. Notebook merge conflicts are painful.
- Never commit data, images, .npy, checkpoints (the .gitignore blocks most of this).
- Save your executed Colab notebook (File > Download .ipynb) into `executed_notebooks/<name>_run.ipynb`. Don't overwrite the clean notebook in `notebooks/` with it.
- Don't use "Save in GitHub" from Colab. Edit code locally and push.

Colab gotchas:
- After pulling new code, restart the Colab session (Runtime > Restart session). autoreload doesn't work on Colab's Python 3.13.
- "can only test a child process" DataLoader messages on Python 3.13 are harmless.
- "Unauthenticated requests to the HF Hub" warning is harmless.
- The CLIP load report lists text_model keys as UNEXPECTED. That's fine, we only use the image half.
- Free Colab GPU time is limited. If no GPU is available, try again later.

## Code conventions

- Put reusable code in `src/`, keep notebooks thin (import from src, short markdown, show outputs).
- All paths, sizes and hyperparameters go in `configs/config.yaml`, read with `src.config`.
- Fixed seed 42, everything resumable (skip finished work, save checkpoints) because Colab disconnects.
- Add a small pytest for new src functions when it's easy.
- Every notebook starts with the same setup cell (copy it from an existing notebook) and ends with a "Findings" markdown cell and the flush cell.
- Comments: short, few words, student style. No long docstrings, no `# =====` banners, no em dashes.
- Markdown cells: a header plus one short line. Cite papers/code where used ("Refs: ...").
- Same evaluation for everything: AUROC per generator, accuracy, F1, real vs fake accuracy, ROC curves, confusion matrices (`src/evaluate.py`).
- Don't invent facts about the data. If something needs checking, write code that checks it and prints the result.
- Labels: `y = 1` = fake (`ai`), `y = 0` = real (`nature`). Model scores are P(fake).
- CSV paths are relative to `genimage_project/` (`raw_relpath`, `processed_relpath`). Read project CSVs with `src.manifest.read_csv` (keeps md5/phash/class ids as strings).
- Never copy or extract a whole GenImage generator archive. Archives are read in place with `7z l -slt` / `7z x -i@list`.
- Dependencies: `requirements.txt` for local, `requirements-colab.txt` on Colab (installs only missing packages, never reinstalls Colab's torch/numpy/pandas).
- Tests: `python -m pytest -q` must pass locally without data.

## References

- Zhu et al. 2023, GenImage, arXiv 2306.08571
- Grommelt et al. 2024, Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets, arXiv 2403.17608
- Ojha et al. 2023, UnivFD (frozen CLIP + linear probe), arXiv 2302.10174
- Wang et al. 2020, CNN-generated images are surprisingly easy to spot... for now, arXiv 1912.11035
- Frank et al. 2020, frequency analysis for fake image detection, arXiv 2003.08685
- He et al. 2016 ResNet (1512.03385), Radford et al. 2021 CLIP (2103.00020), Oquab et al. 2023 DINOv2 (2304.07193)
