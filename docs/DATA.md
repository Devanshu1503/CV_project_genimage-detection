# Data: GenImage subset, sampling, bias, preprocessing

## 1. Dataset

**GenImage** (Zhu et al., *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image*,
NeurIPS 2023 Datasets & Benchmarks, [arXiv 2306.08571](https://arxiv.org/abs/2306.08571),
[GitHub](https://github.com/GenImage-Dataset/GenImage)). It pairs real ImageNet images with images
generated from ImageNet class prompts by 8 generators:

| Folder on Drive | Display name | Type |
|---|---|---|
| `stable_diffusion_v_1_4` | SD1.4 (**training generator**) | latent diffusion |
| `stable_diffusion_v_1_5` | SD1.5 | latent diffusion |
| `wukong` | Wukong | latent diffusion |
| `Midjourney` | Midjourney | commercial diffusion (closed) |
| `ADM` | ADM | pixel-space diffusion |
| `glide` | GLIDE | pixel-space diffusion |
| `VQDM` | VQDM | vector-quantised diffusion |
| `BigGAN` | BigGAN | GAN |

**Storage format.** Each generator is a split ZIP: parts `<base>.z01 … <base>.zNN` (3 GB each) plus a
final `<base>.zip` holding the central directory (SD1.4 alone is ~90 GB). Base names differ per
generator and are **discovered** at runtime (`archive_utils.discover_archives`).

**Inner layout** (official README; **verified at runtime** in notebook 01, steps 2 and 6):
`[<top folder>/]{train,val}/{ai,nature}/<file>`, where `ai` = generated (label 1) and `nature` = real ImageNet (label 0).
Anything that doesn't match is reported as `unknown`, never guessed.

> ⚠️ Fill in from notebook 01 outputs: actual top folders, file counts per split/label, extensions,
> filename patterns, whether ImageNet classes could be parsed.

## 2. Access strategy (why no full extraction)

Colab's disk (~80 GB) can't hold even one generator, and our Drive has ~10 GB free. So:

1. `7z l -slt <base>.zip` lists the archive **in place** on the Drive mount (it reads only the central
   directory). The text output is parsed into a manifest (`manifests/manifest_<generator>.csv.gz`).
2. We sample from the manifests.
3. `7z x <base>.zip -i@<list>` extracts **only** the sampled files to local disk, in batches, with several 7z processes in parallel (`seven_zip.extract_workers`). Optionally a whole
   generator's archive is first copied to the Colab disk (`seven_zip.copy_local_generators`, only if it fits with
   `local_copy_margin_gb` to spare) and deleted afterwards, since random reads from Drive are slow (~1.5 s/image
   measured with a single 7z process). Metadata
   is computed on the local copy, then the **original bytes** are copied to `raw_subset/` on Drive.

Manifests are stored gzip-compressed (`.csv.gz`, read transparently by pandas) because they hold
hundreds of thousands of rows per generator.

## 3. Sampling plan (config: `sampling.*`, seed 42)

| Set | Source | Size |
|---|---|---|
| Training pool | SD1.4, archive split `train` | 5000 ai + 5000 nature |
| Test set ×8 | each generator, archive split `val` | 500 ai + 500 nature |

- **Stratified by ImageNet class** where the filename has a class id (round-robin over classes, so
  as many classes as possible are covered). Coverage is reported per set (notebook 01 step 8).
- The training pool is split **80/20 into train/val** (stratified by label) in notebook 02. *val* is used
  for early stopping and for choosing the linear probe's C. It is never used as a test set.
- SD1.4's own `val` sample is the **in-distribution** test set. The other 7 are **unseen generators**.
- Before extracting, the total size is estimated from the manifest and compared with
  `sampling.drive_budget_gb` (default 6 GB). Over budget → the notebook stops and suggests smaller sizes.

## 4. Leakage controls

Real images come from ImageNet, and the same real photo may appear in several generators' folders.
To keep test sets honest:

1. **Filename check before sampling:** any real image whose filename is in the training sample is
   removed from every test pool (notebook 01 step 9). Notebook 01 step 7 reports how often real
   filenames repeat across generators and across train/val.
2. **Content check after extraction:** md5 (exact duplicates) and a 64-bit DCT perceptual hash
   (custom implementation, `image_utils.phash`) with Hamming distance ≤ 4 (near duplicates). Any test image
   that duplicates a training image is dropped and logged in `manifests/dropped_files.csv`.
   Duplicates *between test sets* are only reported (they don't leak training information).

## 5. Known bias: format and size shortcuts

Grommelt et al. 2024 (*Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets*,
[arXiv 2403.17608](https://arxiv.org/abs/2403.17608)) show that in GenImage:
- real images are **JPEG** (mostly quality ~96) with **varying sizes**,
- fakes are **PNG** at a **fixed size per generator** (reported: Midjourney 1024, SD/Wukong 512,
  ADM/GLIDE/VQDM 256, BigGAN 128),

so a detector can reach high accuracy by learning "JPEG artifacts = real" or "size = fake" instead of
generation artifacts, and then fails on JPEG-compressed fakes.

**We measure it ourselves** (notebook 02 steps 3, 4 and 10): format × label tables, JPEG-quality
histogram (quality estimated from the quantisation tables, `image_utils.estimate_jpeg_quality`), size
plots, and a **metadata-only shortcut test**. Logistic regression and a depth-3 tree predict real vs
fake from `is_jpeg, width, height, file_bytes, jpeg_quality` only, with no pixels. A high score proves
the shortcut exists.

> ⚠️ Fill in: measured format split, JPEG quality of real images, sizes per generator, shortcut-test accuracy/AUROC.

## 6. Preprocessing (config: `preprocess.*`)

Identical for **every** image, real and fake (`image_utils.preprocess_image` + `save_jpeg`):

1. Decode, apply EXIF orientation, convert to **RGB** (transparent images composited on white).
2. **Center-crop** to a square on the shorter side. This removes aspect-ratio differences without distortion.
3. **Resize to 224×224, bicubic.** This removes size differences; 224 is the native input of all three backbones.
4. Save as **JPEG quality 96**, 4:2:0 chroma subsampling, no EXIF/ICC. This removes the format
   difference. q96 matches the dominant quality of the real images, so real images change as little
   as possible.

Output: `processed/<generator>/<role>/<label>/<name>.jpg` + `manifests/processed_index.csv`.
**Verification** (notebook 02 step 14): the shortcut test is rerun on the processed files. Format,
size and quality become constant, so accuracy should fall to ~50%. `file_bytes` can keep some signal
because JPEG size depends on image content (smoothness), so the test is reported with and without it.

**Training-time choices** (notebook 03): ResNet-50 uses random crop (train) / center crop (eval) of
192 px **without resizing**, plus horizontal flips. Resizing would blur generation artifacts
(following Wang et al. 2020, arXiv 1912.11035). No JPEG/blur augmentation is used in Problem 1.

## 7. Limitations

- **Double compression of real images:** reals are JPEG-compressed twice (ImageNet + ours), fakes once.
  Double-JPEG traces could become a new, weaker shortcut.
- **Different resampling per generator:** BigGAN (128 px) is upscaled ×1.75, Midjourney (1024 px) is downscaled
  ~×4.6, SD/Wukong ~×2.3, ADM/GLIDE/VQDM slightly. Up-sampling blurs; strong down-sampling
  removes high-frequency artifacts. Per-generator results partly reflect this.
- Center crop drops borders of non-square images (mostly real ones).
- ImageNet class balance is only as good as the class ids that can be parsed from filenames.
- A subset (10k train, 1k per test generator) gives AUROC estimates with roughly ±0.01–0.02
  uncertainty. Bootstrap confidence intervals are a listed next step.

## 8. Files produced (all under `genimage_project/`)

| File | Content |
|---|---|
| `manifests/archive_inventory.csv` | archives found per generator, parts, size, status |
| `manifests/manifest_<gen>.csv.gz`, `manifest_all.csv.gz` | every file in every archive |
| `manifests/sampling_plan.csv` | the sample before extraction |
| `manifests/raw_metadata*.csv` | format, size, mode, bytes, JPEG quality, md5, pHash per file |
| `manifests/duplicate_pairs.csv`, `dropped_files.csv` | duplicates found, files removed and why |
| `manifests/subset_index.csv` | **source of truth** for the raw subset |
| `manifests/processed_index.csv`, `splits.csv` | processed files + train/val/test assignment |
| `results/*.json, *.csv` | shortcut tests, leakage summary, model results |
| `figures/*.png` | every plot from the notebooks |
| `logs/*.log` | run logs |
