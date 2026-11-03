# Detecting AI-generated images: cross-generator generalisation on GenImage

UChicago MSADS · Advanced Computer Vision (ADSP 31023) · Team project

**Team:** _Name 1 · Name 2 · Name 3 · Name 4_ (fill in)

## Overview

A detector trained on one image generator usually fails on the next one. We study this on
**GenImage** (Zhu et al., NeurIPS 2023, [arXiv 2306.08571](https://arxiv.org/abs/2306.08571)): real
ImageNet photos plus fakes from 8 generators.

| | Problem | Competing models | Main metric |
|---|---|---|---|
| **1** | **Cross-generator detection.** Train real-vs-fake on Stable Diffusion v1.4 only, test on all 8 generators (7 unseen) | fine-tuned **ResNet-50** (champion) vs frozen **CLIP ViT-L/14** / **DINOv2 ViT-B/14** features + linear probe (challengers) | AUROC per generator (+ accuracy, F1, confusion matrices, ROC) |
| **2** | **Robustness** (next phase). JPEG compression / resizing, new SDXL-Turbo fakes, retraining with augmentation | 2 competing models (see [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md)) | AUROC drop under degradation |

**Known dataset bias.** In GenImage, real images are ImageNet JPEGs of varying size while fakes
are PNGs at a fixed size per generator. A model can "cheat" by learning file format or size
(Grommelt et al. 2024, *Fake or JPEG?*, [arXiv 2403.17608](https://arxiv.org/abs/2403.17608)).
Notebook 02 **measures** this bias and **removes** it by re-encoding every image the same way.
See [docs/DATA.md](docs/DATA.md).

## Repository layout

```
configs/config.yaml   ALL paths, seeds, sample sizes, preprocessing and training settings
src/                  importable code (notebooks stay thin)
  config.py           load config, detect Colab vs local, resolve paths, logger
  colab_setup.py      mount Drive, install deps (+ apt p7zip-full), seeds, GPU info
  archive_utils.py    discover split zips, `7z l -slt` parsing, selective `7z x -i@list`
  manifest.py         manifests, path parsing, sampling, leakage / duplicate checks
  image_utils.py      metadata, JPEG-quality estimate, md5 + pHash, preprocessing, FFT
  eda_utils.py        plots, pixel statistics, spectra, metadata shortcut test
  datasets.py         PyTorch Dataset + transforms
  models.py           ResNet-50, CLIP and DINOv2 feature extractors, feature caching
  train.py            training loop (AMP, early stopping, resume), linear probe
  evaluate.py         metrics, per-generator table, ROC / confusion / bar plots
tests/                pytest suite that runs locally without data
notebooks/            01 → 02 → 03, run on Google Colab
docs/                 DATA.md (dataset & preprocessing decisions), NEXT_STEPS.md (handoff)
```

## Data access (one-time, per person)

The raw data is never copied in full. GenImage (~hundreds of GB) stays as split zips on Drive and is
read **in place** with 7-Zip.

1. **GenImage archives.** Make the GenImage folder (sub-folders `ADM`, `BigGAN`, `glide`, `Midjourney`,
   `stable_diffusion_v_1_4`, `stable_diffusion_v_1_5`, `VQDM`, `wukong`) available as
   `MyDrive/genimage`: right-click the shared folder → *Organize → Add shortcut* → My Drive, and name
   the shortcut `genimage`.
2. **Project outputs.** Notebook 01 creates `MyDrive/genimage_project/` (manifests, raw subset,
   processed images, features, checkpoints, results, figures, logs; about 5–6 GB).
   **The person who ran notebook 01 shares `genimage_project` with teammates** (Share → Editor).
   Each teammate then adds a shortcut to it in *their* My Drive, named exactly `genimage_project`.
   After that, teammates can start directly at notebook 02 or 03.
3. Paths are set in [configs/config.yaml](configs/config.yaml) (`paths.colab`). Change them there if
   your shortcuts have other names.

## How to run (Google Colab, in order)

First, put your GitHub repo URL in the `REPO_URL` line of the **first cell** of each notebook
(or open the notebook from GitHub in Colab: *File → Open notebook → GitHub*). If the repo is private,
clone with a personal access token or make it public.

| # | Notebook | Runtime | What it does | Main outputs |
|---|---|---|---|---|
| 1 | [01_build_dataset](notebooks/01_build_dataset.ipynb) | CPU | list archives in place, sample, extract only the sample, dedup/leakage checks | `manifests/subset_index.csv`, `raw_subset/` |
| 2 | [02_eda_and_preprocess](notebooks/02_eda_and_preprocess.ipynb) | CPU | EDA + bias measurement, shared preprocessing, train/val/test split | `processed/`, `manifests/processed_index.csv`, `manifests/splits.csv`, `figures/02_*` |
| 3 | [03_problem1_baselines](notebooks/03_problem1_baselines.ipynb) | **T4 GPU** | CLIP/DINOv2 linear probes vs fine-tuned ResNet-50, per-generator evaluation | `results/problem1_*.csv`, `checkpoints/`, `figures/03_*` |

- **Every notebook starts with the same setup cell.** It clones or pulls the repo, mounts Drive,
  installs dependencies (including `p7zip-full`), sets seeds and prints GPU info.
- **Notebook 01:** run it once with the default `RUN_EXTRACTION = False`. Check the size estimate
  (step 10), then set `RUN_EXTRACTION = True` and run all again.
- **Notebook 03:** switch the runtime first with *Runtime → Change runtime type → T4 GPU*.
- **Everything is resumable.** After a Colab disconnect, just *Run all* again; finished work is skipped,
  and ResNet training resumes from `checkpoints/resnet50/last.pt`.

## Local development (no data needed)

```bash
pip install -r requirements.txt
python -m pytest -q          # tests for parsing, sampling, preprocessing, metrics, training smoke test
```

## Main references

- Zhu et al. 2023, *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image*, arXiv 2306.08571
- Grommelt et al. 2024, *Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets*, arXiv 2403.17608
- Ojha et al. 2023, *Towards Universal Fake Image Detectors that Generalize Across Generative Models* (UnivFD), arXiv 2302.10174
- Wang et al. 2020, *CNN-generated images are surprisingly easy to spot... for now*, arXiv 1912.11035
- He et al. 2016 (ResNet), arXiv 1512.03385 · Radford et al. 2021 (CLIP), arXiv 2103.00020 · Oquab et al. 2023 (DINOv2), arXiv 2304.07193
- Frank et al. 2020, *Leveraging Frequency Analysis for Deep Fake Image Recognition*, arXiv 2003.08685
- Code/weights: [torchvision](https://pytorch.org/vision/stable/models.html), [HF openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14), [HF facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base), [7-Zip / p7zip](https://www.7-zip.org/)
