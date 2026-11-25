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

## Notebook 04: Grad-CAM and confidence intervals

Bootstrap 95% confidence intervals for the notebook 03 AUROCs (1000 resamples, reals and fakes resampled separately).

| Model | SD1.4 (seen) | Mean unseen (7) |
|---|---|---|
| ResNet-50 | 0.991 [0.985, 0.995] | 0.728 [0.716, 0.738] |
| CLIP ViT-L/14 + LR | 0.985 [0.978, 0.990] | 0.839 [0.830, 0.848] |
| DINOv2 ViT-B/14 + LR | 0.935 [0.919, 0.948] | 0.707 [0.696, 0.718] |

- Intervals are about ±0.01 on SD1.4, SD1.5 and Wukong, and up to ±0.035 on ADM, GLIDE, VQDM and BigGAN.
- The mean unseen intervals don't overlap, so the Problem 1 ranking is not test-set noise.
- BigGAN is below 0.5 for the whole interval for ResNet-50 (0.424 [0.391, 0.462]) and DINOv2 (0.397 [0.362, 0.433]). These models rank BigGAN fakes as more real than real photos.
- Paired tests (same resampled images for both models), CLIP minus ResNet-50:

  | Generator | Difference [95% CI] |
  |---|---|
  | SD1.4 (seen) | -0.006 [-0.013, 0.001], tied |
  | SD1.5 | -0.007, ResNet better |
  | Wukong | -0.043, ResNet better |
  | Midjourney | +0.097 |
  | VQDM | +0.122 |
  | ADM | +0.133 |
  | BigGAN | +0.222 |
  | GLIDE | +0.256 |
  | Mean unseen | +0.111 [0.100, 0.123] |

- CLIP beats DINOv2 on all 8 generators. ResNet-50 vs DINOv2 is mixed: ResNet is ahead on mean unseen (+0.020), DINOv2 on Midjourney and GLIDE.
- Grad-CAM (our own implementation, on `layer4[-1]`):
  - Fakes the ResNet catches (SD1.4, SD1.5, Wukong, Midjourney) give one compact blob in the middle of the image.
  - Missed ADM, GLIDE, VQDM and BigGAN fakes look like real photos to it, with no concentrated "fake" evidence at all.
  - 2 of the 4 most confident real-called-fake errors on SD1.4 are smooth surfaces (a yellow bowl, a cup of coffee). Smooth clean texture seems to be part of what it learned as "fake".
- Deletion test (blank 20% of a caught fake):
  - Blanking the heatmap region lowers P(fake) about 3x more than blanking a random box of the same size on the SD family (SD1.4 0.32 vs 0.11), and 2x on Midjourney. So the heatmaps point at what the model uses.
  - But SD1.4 fakes still score 0.66 with that region blanked. Most of the evidence is spread over the whole image, which fits the high-frequency finding from notebook 02.
  - Blanking 20% scattered single pixels instead drops SD1.4 fakes from 0.98 to 0.12. The ResNet is very sensitive to high-frequency disruption.

## Notebook 06: degradation tests

We scored the 3 Problem 1 models on 14 versions of the test sets: clean, JPEG quality 95/85/75/60/50, resize 0.5/0.75/1.5 (then back to 224), blur 1/2, noise 5/10, and "social" (resize 0.75 + JPEG 75). Real and fake images get the same damage. No retraining.

- The clean version matches notebook 03 within 0.0001 AUROC.
- AUROC on SD1.4 (seen) stays above 0.89 for every model and condition. ResNet-50 goes from 0.991 to 0.950 at JPEG 50.
- But the 0.5 cutoff breaks. Damage pushes every score towards "real":

  | SD1.4 fake accuracy | Clean | JPEG 50 | Blur 2 | Noise 10 |
  |---|---|---|---|---|
  | ResNet-50 | 0.96 | 0.48 | 0.26 | 0.13 |
  | CLIP + LR | 0.95 | 0.48 | 0.40 | 0.28 |

  Real accuracy stays at 0.95-1.0. So one JPEG re-save hides about half the fakes, while real photos are still called real.
- Mean unseen AUROC:

  | Model | Clean | JPEG 85 | JPEG 50 | Resize 1.5 | Blur 2 | Noise 10 | Social |
  |---|---|---|---|---|---|---|---|
  | ResNet-50 | 0.728 | 0.729 | 0.688 | 0.619 | 0.650 | 0.727 | 0.681 |
  | CLIP + LR | 0.839 | 0.768 | 0.732 | 0.779 | 0.719 | 0.764 | 0.730 |
  | DINOv2 + LR | 0.707 | 0.674 | 0.680 | 0.692 | 0.684 | 0.621 | 0.681 |

- CLIP is still best on unseen generators under every condition, but it loses the most. Its lead over ResNet-50 shrinks from +0.111 to about +0.04. Most of the drop already happens at JPEG 85.
- ResNet-50 is hurt most by enlarging (resize 1.5: ADM 0.46, GLIDE 0.41, BigGAN 0.26). Upscaling smooths away the fine detail it relies on.
- DINOv2 changes least but starts lowest. Noise is its weak spot.
- ResNet-50 on BigGAN goes from 0.42 to 0.68 with noise 10. Noise removes the detail that made BigGAN look "extra real" to it.
