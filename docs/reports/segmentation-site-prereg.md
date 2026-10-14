# Plant outline on the site: pre-registration (U-Net / ResNet-34)

Written 2026-09-30, **before** the U-Net below was trained or scored.

## Why a U-Net

The tuned segmenter is SegFormer-B0 (field-eval IoU 0.6129, five-seed mean at
threshold 0.5; [field-compare.md](field-compare.md)). Its encoder weights come
from NVIDIA's SegFormer, whose licence (section 3.3) allows use "for research
or evaluation purposes only". A public identification site is not clearly
either, so SegFormer weights stay off the site. The U-Net / ResNet-34 has no
such clause, but no U-Net was ever trained with the field recipe, so its field
quality is unknown.

## What is trained

`scripts/hellbender_field_unet.slurm`: the `field` arm of
`hellbender_field_compare.slurm` with `--model unet-resnet34` and every other
flag unchanged (174 curated + field images, 1024 px, 80 epochs, patience 15,
learning rate 3e-4, BCE + Tversky, batch 8, no growth-form balancing), seeds
17, 101, 202, 303, 404. No other recipe is tried.

## Decision rule

1. **Quality bar.** Score all five seeds on the 17 field-eval images with the
   same preprocessing as `scripts/score_field_compare.py` (whole image resized
   to 1024 x 1024, bilinear; ImageNet normalisation). The U-Net ships only if
   its five-seed mean IoU at threshold 0.5 is **>= 0.593** (SegFormer's 0.6129
   minus 0.02). Paired per-seed differences against SegFormer are reported but
   don't decide.
2. **Which seed ships:** the one with the highest curated-validation IoU
   (`best_iou` in its metrics file), not the best on field-eval. Ties go to the
   lower seed.
3. **Threshold:** chosen as it was for SegFormer, the threshold in
   {0.4, 0.5, 0.6, 0.7, 0.8} with the lowest median absolute area error on
   field-eval over the five seeds. This tunes on field-eval, as the SegFormer
   choice did, and the report says so.
4. **Browser model.** Export the shipped seed to ONNX with the normalisation
   inside the graph. A smaller variant (fp16 or int8) replaces fp32 only if its
   mask agrees with PyTorch's at IoU >= 0.99 on every one of the 19 validation
   and 17 field-eval images. The same bar applies to Python ONNX Runtime
   against PyTorch, and to the browser against Python ONNX Runtime.
5. **Speed** is measured, not a gate: if an outline takes more than about 10 s
   on a server core in WASM, the site runs it only when the user asks
   ("Show outline"), with a progress message.

If the quality bar fails, the site launches without an outline. SegFormer-B0 is
not used as a fallback.
