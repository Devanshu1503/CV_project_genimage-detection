# Findings so far

Results from our Colab runs. The executed notebooks with all outputs are in [executed_notebooks/](../executed_notebooks/).

## Notebook 01: building the subset

- We read the full GenImage dataset (656 GB of split zips on Drive) in place with 7z, without extracting it.
- We sampled 18,000 images, 4.91 GB in total:
  - SD1.4 train pool: 5000 real + 5000 fake
  - test: 500 real + 500 fake for each of the 8 generators
- 0 corrupt files and 0 dropped files.
- 0 real filenames are shared across generators or between train and val, so we found no filename leakage.
- We found 2 near-duplicate pairs, both between SD1.4 and SD1.5 test fakes. They are probably the same prompt and seed, since SD1.5 is fine-tuned from SD1.4. Both are test images, so this isn't leakage and we kept them.
- Extraction from Drive took about 0.5 s/image with 4-6 parallel 7z workers, vs 1.4 s/image with 1 worker.
- When we deleted a Colab runtime mid-run, Drive sync lost some writes and the next session made a duplicate `raw_subset/` folder. We fixed it with `flush_drive()` at the end of every notebook and a rerun. The extraction is resumable, so the rerun only did the missing files.

## Notebook 02: EDA and preprocessing

- Format bias: all 9000 real images are JPEG (median quality 96) and all 9000 fakes are PNG.
- Size bias: every generator outputs one fixed size, while real images are mostly 500x375.

  | Generator | Size |
  |---|---|
  | SD1.4, SD1.5, Wukong | 512 |
  | Midjourney | 1024 |
  | ADM, GLIDE, VQDM | 256 |
  | BigGAN | 128 |

- Shortcut test on metadata only: a decision tree learned the rule "is_jpeg means real" and got 100% accuracy without looking at a single pixel.
- After our preprocessing (center crop, resize to 224, JPEG q96 for every image), the same test drops to 50%, which is chance.
  - With file size added it gets 52-56%. That's content-driven, because JPEG size depends on how smooth the image is.
- Pixel statistics: brightness differs only a little between real and fake (KS D = 0.079).
- Frequency spectra:
  - BigGAN shows a grid of FFT peaks, which are GAN upsampling artifacts.
  - GLIDE is smoother than real images.
  - SD1.4, SD1.5 and Wukong have extra high-frequency energy, and it survives preprocessing.
- Outliers: 149 tiny real images, 170 grayscale real images and 17 grayscale Midjourney images. We kept all of them.
- Splits:
  - train 4000 + 4000 (SD1.4)
  - val 1000 + 1000 (SD1.4)
  - test 500 + 500 for each of the 8 generators

## Notebook 03: Problem 1 baselines

In progress.
