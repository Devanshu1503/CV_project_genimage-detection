# Detecting AI-generated images on GenImage

UChicago MSADS, Advanced Computer Vision (ADSP 31023), team project.

## What this is

We want to know if a fake-image detector trained on one generator still works on generators it has never seen. We train on Stable Diffusion 1.4 images only and test on all 8 generators in GenImage. We compare a fine-tuned ResNet-50 against frozen CLIP and DINOv2 features with a linear classifier on top, using AUROC per generator.

Problem 2 (next) is about robustness: JPEG compression, resizing and new generators.

## Data and why we sample it

GenImage (Zhu et al. 2023) has real ImageNet photos plus fakes from 8 generators: SD1.4, SD1.5, Midjourney, Wukong, ADM, GLIDE, VQDM and BigGAN. It's about 656 GB of split zip files. That doesn't fit on Colab's disk or in our Drive, so we leave the zips on Drive and use 7z to pull out only the files we sample:

- SD1.4 train pool: 5000 real + 5000 fake
- test: 500 real + 500 fake for each of the 8 generators

That's 18,000 images, about 5 GB. We also check for leakage between train and test by filename and by image hash. Details are in [docs/DATA.md](docs/DATA.md).

## The bias we found

In GenImage every real image is a JPEG and every fake is a PNG at a fixed size per generator. This was pointed out by Grommelt et al. 2024. A model can just learn "JPEG means real" and never look at the actual image.

We checked it ourselves. A small decision tree that only sees file metadata (format, width, height, file size, JPEG quality) gets 100% accuracy. So we preprocess every image the same way: RGB, center crop, resize to 224, save as JPEG quality 96. After that the same test drops to 50%, which is chance.

## How to run

All notebooks run on Google Colab. The first cell of each one does the setup:

- clones or pulls this repo
- mounts Drive
- installs only the packages that are missing (it doesn't touch Colab's torch, numpy and so on)
- sets the random seeds

One-time Drive setup:

1. Add a shortcut to the shared GenImage folder in your My Drive and name it `genimage`.
2. Notebook 01 writes everything to `MyDrive/genimage_project/`. Whoever ran it shares that folder with the team. Everyone else adds a shortcut to it named `genimage_project`, and can then start at notebook 02 or 03.

Paths are in [configs/config.yaml](configs/config.yaml) if your shortcuts have other names.

| Notebook | Runtime | What it does |
|---|---|---|
| [01_build_dataset](notebooks/01_build_dataset.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Devanshu1503/CV_project_genimage-detection/blob/main/notebooks/01_build_dataset.ipynb) | CPU | lists the zips, samples, extracts the sample, checks for duplicates and leakage |
| [02_eda_and_preprocess](notebooks/02_eda_and_preprocess.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Devanshu1503/CV_project_genimage-detection/blob/main/notebooks/02_eda_and_preprocess.ipynb) | CPU | EDA, bias checks, preprocessing, train/val/test split |
| [03_problem1_baselines](notebooks/03_problem1_baselines.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Devanshu1503/CV_project_genimage-detection/blob/main/notebooks/03_problem1_baselines.ipynb) | T4 GPU | CLIP and DINOv2 linear probes vs ResNet-50, results per generator |

For notebook 03, pick Runtime > Change runtime type > T4 GPU first.

Everything is resumable. If Colab disconnects, run all again and finished steps get skipped. ResNet training continues from its last checkpoint.

## Colab + Drive rules

Colab uploads Drive writes in the background, and a new session can miss files the old one wrote.

- Don't use "Disconnect and delete runtime" while files are being written to Drive.
- If something hangs, use Runtime > Interrupt execution instead.
- Run the last cell of each notebook ("Run before closing the tab") before you close the tab.
- If files seem to be missing, open Drive in the browser and look for duplicate folders with the same name (for example two `raw_subset/` folders). Merge them by hand, because Colab can't see the duplicates.

## Repo layout

```
configs/config.yaml   all paths, sample sizes and hyperparameters
src/                  the actual code, the notebooks just call it
notebooks/            01, 02, 03 (clean, no outputs)
executed_notebooks/   our Colab runs with outputs
tests/                pytest, runs locally without the data
docs/                 DATA.md, FINDINGS.md, NEXT_STEPS.md
```

To run the tests locally: `pip install -r requirements.txt`, then `python -m pytest -q`.

## Status

- Notebook 01 (dataset) and notebook 02 (EDA + preprocessing) are done. Results are in [docs/FINDINGS.md](docs/FINDINGS.md).
- Notebook 03 (Problem 1 baselines) is in progress.
- Next up: Problem 2 (robustness) and the MLOps part. See [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md).

## Team

- Mahima Goyal
- Devanshu Khadka
- Sadaf Khan
- Sameeksha Kini
- Anuradha Rakh

## References

- Zhu et al. 2023, GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image, arXiv 2306.08571
- Grommelt et al. 2024, Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets, arXiv 2403.17608
- Ojha et al. 2023, Towards Universal Fake Image Detectors that Generalize Across Generative Models, arXiv 2302.10174
- Wang et al. 2020, CNN-generated images are surprisingly easy to spot... for now, arXiv 1912.11035
- Frank et al. 2020, Leveraging Frequency Analysis for Deep Fake Image Recognition, arXiv 2003.08685
- He et al. 2016 (ResNet), arXiv 1512.03385
- Radford et al. 2021 (CLIP), arXiv 2103.00020
- Oquab et al. 2023 (DINOv2), arXiv 2304.07193
