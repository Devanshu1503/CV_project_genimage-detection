# Data: GenImage subset, sampling, bias, preprocessing

## 1. Dataset

GenImage (Zhu et al., GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image,
NeurIPS 2023 Datasets & Benchmarks, [arXiv 2306.08571](https://arxiv.org/abs/2306.08571),
[GitHub](https://github.com/GenImage-Dataset/GenImage)) pairs real ImageNet images with images
generated from ImageNet class prompts by 8 generators:

| Folder on Drive | Name | Type |
|---|---|---|
| `stable_diffusion_v_1_4` | SD1.4 (we train on this one) | latent diffusion |
| `stable_diffusion_v_1_5` | SD1.5 | latent diffusion |
| `wukong` | Wukong | latent diffusion |
| `Midjourney` | Midjourney | commercial diffusion (closed) |
| `ADM` | ADM | pixel-space diffusion |
| `glide` | GLIDE | pixel-space diffusion |
| `VQDM` | VQDM | vector-quantized diffusion |
| `BigGAN` | BigGAN | GAN |

Storage format: each generator is a split zip, parts `<base>.z01 ... <base>.zNN` (3.2 GB each) plus a
final `<base>.zip` with the central directory. The whole thing is 656 GB, SD1.4 alone is 96 GB. Base names
differ per generator, so we find them at runtime (`archive_utils.discover_archives`).

Inner layout (official README, checked in notebook 01 steps 2 and 6):
`<top folder>/{train,val}/{ai,nature}/<file>`, where `ai` = generated (label 1) and `nature` = real ImageNet (label 0).
Anything that doesn't match gets reported as `unknown`, we never guess.

What we actually saw in notebook 01:
- One top folder per generator (e.g. `imagenet_ai_0419_sdv4`), 2,681,167 files in total, 0 unknown split/label, 0 non-image files.
- Train has ~162k files per label, val 6,000 per label (8,000 for SD1.5).
- Every `ai` file is `.png`, every `nature` file is `.jpeg`.
- Fake filenames start with a class index (`000_sdv4_00000.png`), except GLIDE and VQDM, which have none.
  Train reals use synset names (`n01440764_10043.JPEG`); val reals are `ILSVRC2012_val_*` without a class.
- SD1.5 has a stray Drive copy `imagenet_ai_0424_sdv5 (1).z26` next to the real parts. It gets flagged and ignored.

## 2. Access strategy (why we don't extract everything)

656 GB doesn't fit on the Colab disk, and our Drive only has ~10 GB free. So:

1. `7z l -slt <base>.zip` lists each archive in place on the Drive mount (it only reads the central
   directory). We parse the output into a manifest (`manifests/manifest_<generator>.csv.gz`).
2. We sample from the manifests.
3. `7z x <base>.zip -i@<list>` extracts only the sampled files to local disk, in batches, with several
   7z processes in parallel (`seven_zip.extract_workers`). Optionally a whole generator's archive is
   copied to the Colab disk first (`seven_zip.copy_local_generators`, only if it fits with
   `local_copy_margin_gb` to spare) and deleted afterwards. We compute metadata on the local copy, then
   copy the original bytes to `raw_subset/` on Drive.

Speed: about 1.4 s/image with one 7z process and about 0.5 s/image with 4-6 in parallel. We ended up using
6 workers and no local copy.

Running notebook 01: step 4b (`RUN_BENCHMARK = True`) times 1/4/8 workers and copying one archive part,
then prints which `extract_workers` / `copy_local_generators` to put in the config. The extraction itself only
runs with `RUN_EXTRACTION = True`. Check the size estimate in step 10 first.

Manifests are saved gzip-compressed (`.csv.gz`, pandas reads them directly) because they have
hundreds of thousands of rows per generator.

## 3. Sampling plan (config: `sampling.*`, seed 42)

| Set | Source | Size |
|---|---|---|
| Training pool | SD1.4, archive split `train` | 5000 ai + 5000 nature |
| Test set x8 | each generator, archive split `val` | 500 ai + 500 nature |

- We stratify by ImageNet class where the filename has a class id (round-robin over classes, so we
  cover as many classes as we can). Notebook 01 step 8 reports coverage per set. In our run the SD1.4
  train sample covers all 1000 classes (5 per class per label) and each test fake set covers 500.
- Notebook 02 splits the training pool 80/20 into train and val, stratified by label. We use val for early
  stopping and for picking the linear probe's C, never as a test set.
- SD1.4's own `val` sample is the in-distribution test set. The other 7 are the unseen generators.
- Before extracting, notebook 01 estimates the total size from the manifest and compares it with
  `sampling.drive_budget_gb` (default 6 GB). If it's over budget, the notebook stops and suggests smaller sizes.
  Our sample came out at 4.91 GB.

## 4. Leakage controls

The real images come from ImageNet, so the same real photo could in principle show up in several generators'
folders. To keep the test sets honest:

1. Filename check before sampling: any real image whose filename is in the training sample is
   removed from every test pool (notebook 01 step 9). Step 7 reports how often real
   filenames repeat across generators and across train/val. In our run that was 0 in both cases.
2. Content check after extraction: md5 for exact duplicates and a 64-bit DCT perceptual hash
   (our own implementation, `image_utils.phash`) with Hamming distance <= 4 for near duplicates. Any test image
   that duplicates a training image gets dropped and logged in `manifests/dropped_files.csv`.
   Duplicates between test sets are only reported, because they don't leak training information.
   We found 2 near-duplicate pairs (SD1.4 vs SD1.5 test fakes) and dropped nothing.

## 5. Known bias: format and size shortcuts

Grommelt et al. 2024 (Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets,
[arXiv 2403.17608](https://arxiv.org/abs/2403.17608)) showed that in GenImage:
- real images are JPEG (mostly quality ~96) with varying sizes,
- fakes are PNG at a fixed size per generator (Midjourney 1024, SD/Wukong 512, ADM/GLIDE/VQDM 256, BigGAN 128).

So a detector can score well by learning "JPEG artifacts = real" or "this size = fake" instead of
actual generation artifacts, and then fail on JPEG-compressed fakes.

We checked this ourselves in notebook 02 (steps 3, 4 and 10): format x label tables, a JPEG-quality
histogram (quality estimated from the quantization tables, `image_utils.estimate_jpeg_quality`), size
plots, and a metadata-only shortcut test. A logistic regression and a depth-3 tree try to predict real vs
fake from just `is_jpeg, width, height, file_bytes, jpeg_quality`, no pixels.

What we measured:
- All 9000 real images are JPEG (median quality 96, mean 92.7), all 9000 fakes are PNG.
- Each generator has exactly one size, matching the list above. Real images are mostly 500x375.
- The tree only needs `is_jpeg` and gets 100% accuracy (AUROC 1.0). Logistic regression gets 99.99%.

## 6. Preprocessing (config: `preprocess.*`)

The same steps for every image, real and fake (`image_utils.preprocess_image` + `save_jpeg`):

1. Decode, apply EXIF orientation, convert to RGB (transparent images go on a white background).
2. Center-crop to a square on the shorter side. This gets rid of aspect-ratio differences without stretching.
3. Resize to 224x224, bicubic. This gets rid of size differences, and 224 is what all three backbones expect.
4. Save as JPEG quality 96, 4:2:0 chroma subsampling, no EXIF/ICC. This gets rid of the format
   difference. We picked q96 because most real images already are q96, so they change as little as possible.

Output: `processed/<generator>/<role>/<label>/<name>.jpg` + `manifests/processed_index.csv`.

Check (notebook 02 step 14): we rerun the shortcut test on the processed files. Format, size and quality are
now constant, and accuracy drops to 50%. `file_bytes` still carries a bit of signal (52-56%) because JPEG size
depends on image content (how smooth it is), so we report the test with and without it.

Training-time choices (notebook 03): ResNet-50 uses a random crop (train) or center crop (eval) of
192 px without resizing, plus horizontal flips. Resizing would blur the generation artifacts
(following Wang et al. 2020, arXiv 1912.11035). Problem 1 uses no JPEG or blur augmentation.

## 7. Limitations

- Double compression of real images: reals get JPEG-compressed twice (ImageNet + us), fakes only once.
  Double-JPEG traces could turn into a new, weaker shortcut.
- Different resampling per generator: BigGAN (128 px) gets upscaled x1.75, Midjourney (1024 px) downscaled
  ~x4.6, SD/Wukong ~x2.3, ADM/GLIDE/VQDM only slightly. Upscaling blurs, and strong downscaling
  removes high-frequency artifacts, so per-generator results partly reflect this.
- Center crop cuts off the borders of non-square images (mostly real ones).
- Class balance is only as good as the class ids we can parse from filenames (none for GLIDE, VQDM and val reals).
- With 10k training images and 1k per test generator, AUROC estimates have roughly +-0.01-0.02
  uncertainty. Bootstrap confidence intervals are on the to-do list.

## 8. Files produced (all under `genimage_project/`)

| File | Content |
|---|---|
| `manifests/archive_inventory.csv` | archives found per generator, parts, size, status |
| `manifests/manifest_<gen>.csv.gz`, `manifest_all.csv.gz` | every file in every archive |
| `manifests/sampling_plan.csv` | the sample before extraction |
| `manifests/raw_metadata*.csv` | format, size, mode, bytes, JPEG quality, md5, pHash per file |
| `manifests/duplicate_pairs.csv`, `dropped_files.csv` | duplicates found, files removed and why |
| `manifests/subset_index.csv` | the main index of the raw subset, notebooks 02 and 03 read this |
| `manifests/processed_index.csv`, `splits.csv` | processed files + train/val/test assignment |
| `results/*.json, *.csv` | shortcut tests, leakage summary, model results |
| `figures/*.png` | every plot from the notebooks |
| `logs/*.log` | run logs |
