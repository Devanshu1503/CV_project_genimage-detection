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

All three models are trained on SD1.4 only. The table shows AUROC on each test set.

| Generator | ResNet-50 | CLIP ViT-L/14 + LR | DINOv2 ViT-B/14 + LR |
|---|---|---|---|
| SD1.4 (seen) | 0.991 | 0.985 | 0.935 |
| SD1.5 | 0.995 | 0.987 | 0.937 |
| Midjourney | 0.833 | 0.930 | 0.870 |
| Wukong | 0.989 | 0.946 | 0.914 |
| ADM | 0.624 | 0.756 | 0.526 |
| GLIDE | 0.615 | 0.871 | 0.721 |
| VQDM | 0.613 | 0.735 | 0.584 |
| BigGAN | 0.424 | 0.646 | 0.397 |
| Mean unseen (7) | 0.728 | 0.839 | 0.707 |

- On SD1.4 val, ResNet-50 got AUROC 0.993 and accuracy 0.968. Its val AUROC went up every epoch (0.985 -> 0.993), and the best was epoch 5. Each epoch took about 35 s on a T4.
- The linear probes both picked C = 0.01. On SD1.4 val, CLIP got AUROC 0.989 / accuracy 0.942 and DINOv2 got 0.937 / 0.858.
- All three models do well on SD1.4, SD1.5 and Wukong, which are similar latent diffusion models.
- The CLIP challenger generalizes best, beating the ResNet-50 champion by 0.11 mean unseen AUROC. DINOv2 is weakest overall.
- Real-image accuracy stays at 0.93-0.96 on every test set for ResNet and CLIP. The errors are fakes being called real.
- ResNet's accuracy on fakes collapses on unseen generators:

  | Generator | ResNet fake accuracy |
  |---|---|
  | ADM | 0.07 |
  | GLIDE | 0.10 |
  | VQDM | 0.11 |
  | BigGAN | 0.01 |
  | Midjourney | 0.45 |

- BigGAN AUROC is below 0.5 for ResNet and DINOv2. This is likely because the 128 px BigGAN images get upscaled (blurred) and lack the high-frequency traces the models learned from SD.
