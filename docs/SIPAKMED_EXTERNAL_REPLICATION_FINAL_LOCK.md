# SIPaKMeD External Replication Final Lock

Date: 2026-08-13

Status: FINAL OUTCOME-BLIND PROTOCOL LOCK

This document freezes the SIPaKMeD external replication protocol before any
SIPaKMeD model outcome is inspected. It is data-only/protocol-only. No model
checkpoint was loaded for SIPaKMeD inference, no SIPaKMeD inference was run, and
no SIPaKMeD prediction metrics were calculated while creating this lock.

## Dataset Lock

Preflight GO commit:

`ab66b33314d23fda686e236150c354a915f21a7f`

Frozen eligibility manifest:

`results/sipakmed_preflight/sipakmed_eligibility_manifest.csv`

Manifest SHA256:

`e86e99441866fb03d457e3867032ecee4e6792285c6647d93e35089b923de1cd`

The exact eligible SIPaKMeD sample set is frozen as the 4000 manifest rows with
`technical_valid=TRUE`. The exact `sample_id -> parent_source_id` mapping is
frozen by the same manifest columns. The manifest hash above is the binding
record of the eligible sample IDs, excluded sample IDs, contour paths, image
paths, technical validity flags, exclusion reasons, diagnostic labels if
available, and parent-source grouping.

Frozen technical inventory:

- Total isolated-cell images: 4049
- Technically eligible images: 4000
- Unique technically excluded images: 49
- Parent source IDs recovered: 966
- Image mode: RGB for 4049/4049 images
- Image dtype: uint8 for 4049/4049 images
- Cropped nucleus contour files: 4049
- Cropped cytoplasm/whole-cell contour files: 4049
- Duplicate sample IDs: 0
- Duplicate image hash groups: 0
- Duplicate contour hash groups: 0

Official archive integrity was verified in the data-only preflight:

| Archive | Size match | SHA256 |
| --- | --- | --- |
| `im_Dyskeratotic.7z` | YES | `a8768bc03c9e814063ff946b7ef7548067e16b760168267d2b12e95a66955cce` |
| `im_Koilocytotic.7z` | YES | `653578e878e3cc63a8510cd7ec26d918637d9881e913e29a5bb205d0d9cfb70d` |
| `im_Metaplastic.7z` | YES | `97f1badc5f606f127d2af0d18a008ebd41a2b60f44785cb332a6eaa33004d2af` |
| `im_Parabasal.7z` | YES | `13e9c29f016a5b83731e038cfe518671a7815df28836dee706acba6ecd2369a8` |
| `im_Superficial-Intermediate.7z` | YES | `3b73e406df14d60b1bd9c8ac3b01b594e70bb51ba32fb080141fc6a7f489a599` |

Technical exclusion reasons may overlap within the 49 unique excluded samples.
Frozen reason counts:

- `nucleus_contour_out_of_bounds`: 26
- `empty_nucleus_mask`: 25
- `undefined_nucleus_circularity`: 25
- `nucleus_pixels_outside_cell_mask`: 22
- `cytoplasm_contour_out_of_bounds`: 2

Frozen unique exclusion combinations:

- `nucleus_contour_out_of_bounds;empty_nucleus_mask;undefined_nucleus_circularity`: 25
- `nucleus_pixels_outside_cell_mask`: 21
- `cytoplasm_contour_out_of_bounds`: 2
- `nucleus_contour_out_of_bounds;nucleus_pixels_outside_cell_mask`: 1

No additional SIPaKMeD samples may be excluded after model outcomes are known,
except for a true runtime or data-integrity failure that prevents evaluating a
sample exactly as locked. Any such failure must be reported with the manifest
row, reason, timestamp, and no replacement sample.

## Ground-Truth Construction Lock

Official contour format:

Plain text `.dat`, one `x,y` floating-point coordinate pair per line,
comma-separated.

Coordinate convention:

Native image coordinate space, origin at the top-left. `x` is the horizontal
column coordinate and `y` is the vertical row coordinate. Coordinates are
continuous pixel coordinates bounded by `0 <= x < width` and `0 <= y < height`;
they are not integer label indices.

Cytoplasm contour semantics:

The official cytoplasm contour represents the whole-cell outer boundary
including the nucleus. The canonical cytoplasm-only mask is therefore:

`cytoplasm_only_mask = whole_cell_mask AND NOT nucleus_mask`

Frozen 3-class ground-truth construction:

- Class `0`: background/other
- Class `1`: cytoplasm-only pixels
- Class `2`: nucleus pixels

The nucleus class takes precedence over the whole-cell/cytoplasm contour when
constructing the mutually exclusive target. N/C is defined as:

`nucleus area / cytoplasm-only area`

N/C is computable for all 4000 technically eligible samples and is not
computable for all 4049 raw isolated images.

## Model Lock

The external replication uses the existing frozen Herlev baseline checkpoint
family. No checkpoint may be selected, rejected, tuned, averaged, or recalibrated
using SIPaKMeD outcomes.

All three baseline seeds are locked for evaluation:

| Seed | Checkpoint | SHA256 |
| ---: | --- | --- |
| 20260803 | `results/pilot/fair_repeat/checkpoints/baseline_seed_20260803_best_val_dice.pt` | `8c52642452c674bc4b929a7b643c9f69f0fc600d2f7b5eafdf733287e98eed51` |
| 20260804 | `results/pilot/fair_repeat/checkpoints/baseline_seed_20260804_best_val_dice.pt` | `d8bf421f71b0ba899a9d70700f6bdb1155110fd91af8617daff97325aed2b6af` |
| 20260805 | `results/pilot/fair_repeat/checkpoints/baseline_seed_20260805_best_val_dice.pt` | `a8e1563e5dc86f67ed5b393a8f1fc78a96165b727cedd766aa8c1131bc5fdc26` |

Checkpoint selection rule:

Use the existing `best_val_dice` Herlev baseline checkpoint for each locked seed.
Do not inspect SIPaKMeD performance to choose among seeds. Report all seeds and
the seed-family aggregate using the cluster-aware procedure below.

Architecture and training configuration:

- Model: `UNet(in_channels=3, n_classes=3, base_channels=32)`
- Output labels: `0=background/other`, `1=cytoplasm`, `2=nucleus`
- Training data source: Herlev only
- External data used during training: none
- SIPaKMeD fine-tuning/recalibration: prohibited
- SIPaKMeD-specific thresholding, checkpoint selection, or model adaptation:
  prohibited

Locked Herlev pilot configuration:

- `seed`: 20260803
- `image_size`: `[128, 128]`
- `batch_size`: 8
- `learning_rate`: 0.001
- `base_channels`: 32
- `early_stopping_patience`: 10
- `lambda_nc`: 0 for the baseline family

## Preprocessing Lock

Input preprocessing must reuse the Herlev baseline preprocessing:

- Load RGB images only; do not convert to grayscale.
- Convert uint8 `[0, 255]` RGB pixels to float `[0, 1]`.
- Resize isotropically with aspect-ratio preservation.
- Center-pad to `128 x 128` using black pixels.
- Do not crop, stain-normalize, histogram-match, denoise, sharpen, or apply any
  SIPaKMeD-specific preprocessing.
- Restore predictions to native image geometry before morphometric measurement.

The preprocessing lock applies equally to all seeds and all SIPaKMeD samples.

## Primary Analysis Lock

Primary scientific question:

Does the frozen Herlev-trained segmentation model preserve nucleus boundary
morphometry on SIPaKMeD, and is conventional nucleus overlap sufficient to
explain morphometric fidelity under external replication?

Primary unit of measurement:

Cell-level measurements are computed for technically eligible samples, but
uncertainty intervals must cluster-resample by `parent_source_id` to avoid
pseudo-replication.

Primary outcomes:

- Nucleus Dice
- Nucleus circularity absolute error
- Spearman association between nucleus Dice and nucleus circularity absolute
  error

Primary uncertainty procedure:

- Cluster unit: `parent_source_id`
- Bootstrap replicates: 10000
- Bootstrap RNG seed: 20260813
- Resampling: sample parent-source clusters with replacement, include all
  eligible cells from each sampled parent cluster, then compute the estimand.
- Confidence interval: percentile interval from the 10000 cluster-bootstrap
  replicates.

Report effect estimates and confidence intervals. Do not apply a success/failure
threshold and do not reinterpret the protocol based on observed SIPaKMeD
outcomes.

## Secondary Analysis Lock

Secondary outcomes:

- Nucleus area absolute error
- Nucleus perimeter absolute error
- N/C absolute error
- Foreground or cytoplasm/whole-cell overlap, as technically appropriate from
  the restored prediction and frozen ground truth

Secondary association analyses:

- Overlap versus area absolute error
- Overlap versus perimeter absolute error
- Overlap versus N/C absolute error
- Nucleus Dice versus nucleus area absolute error
- Nucleus Dice versus nucleus perimeter absolute error

Use the same parent-source cluster bootstrap with 10000 replicates and RNG seed
20260813 for secondary confidence intervals.

Diagnostic-class summaries may be reported only as descriptive frozen subgroup
summaries. They are not decision strata and may not be used for outcome-driven
exclusions, retuning, or checkpoint choice.

## External Specificity Analysis Lock

The Herlev post-segmentation correction is locked as a secondary external
specificity analysis.

Fixed correction:

- Operation: morphological closing
- Kernel: `3 x 3` elliptical structuring element
- Iterations: 1
- Scope: apply to the predicted nucleus mask only after frozen model prediction
- Tuning: none
- Parameter sweep: prohibited
- Adaptive rules: prohibited

Matched-null control:

Use the existing GT-blind geometry-matched null-control procedure from the
Herlev closing-correction work. The null must be generated without using
SIPaKMeD ground-truth errors or SIPaKMeD outcome metrics. It must match the
prespecified raw-to-closing geometry change according to the existing local
random dual-matched null convention and may not be redesigned after outcomes.

Specificity reporting:

- Raw circularity absolute error
- Closing circularity absolute error
- Matched-null circularity absolute error
- Paired raw-to-closing benefit
- Paired raw-to-null benefit
- Closing-specific residual benefit
- Parent-source cluster-aware confidence intervals

This analysis is secondary and external. It cannot change the primary
replication conclusions, model choice, preprocessing, or eligible sample set.

## Outcome Discipline Lock

Before the final lock:

- SIPaKMeD outcomes inspected: NO
- Checkpoint used for SIPaKMeD inference: NO
- SIPaKMeD inference run: NO
- SIPaKMeD prediction metrics inspected: NO

After this lock, the next permissible action is exactly one frozen SIPaKMeD
external replication run following this document. No exploratory SIPaKMeD
inspection, tuning, seed selection, lambda search, calibration, or preprocessing
adaptation is allowed before that run.

## Provenance Lock

Preflight artifacts:

| Artifact | SHA256 |
| --- | --- |
| `results/sipakmed_preflight/sipakmed_eligibility_manifest.csv` | `e86e99441866fb03d457e3867032ecee4e6792285c6647d93e35089b923de1cd` |
| `results/sipakmed_preflight/preflight_summary.json` | `c359aaaa50b4ed06c8ea086f8e3d5ac8ba6a856d5c0b7e3ec2cc2df9165f9844` |
| `src/data_prep/load_sipakmed.py` | `a20d049aff5f95ce93fe737b5e3649c48a7c5f33d4b7e44bd81941386f5f857a` |
| `experiments/run_sipakmed_preflight.py` | `f7d24a792a4a949dd40a8e8a90beb98f60c49bcc782af6a7b6dccc1e909d5666` |

Relevant implementation files:

| Artifact | SHA256 |
| --- | --- |
| `src/segmentation/unet_model.py` | `39a75c0bbd25881014b2056598de24f13fbb0d8787de1477c9f610a236223d1b` |
| `src/segmentation/train_unet.py` | `87479187ec0bd799a124d49473db1d22752de83e1cad67ebea1b057810271600` |
| `src/measurements/morphometry.py` | `ef551a2e234f8505c38feab210de6e05305df440c1d90aae3041f80dca59d000` |
| `src/postprocessing.py` | `a615273615bdb82493527fe0762ecdaed72f706c2a3fb461f8612449bdcf20cf` |
| `src/closing_correction_confirmatory.py` | `f6fb074d87d21acf0dc8ab0761c1dedc8e1b0b298165d130c19d7120bc8bda93` |
| `src/closing_dual_matched_null.py` | `5232de63f0bd896d943307a47147e0a3b40cc25daab91e36d230a662c140e112` |

Final protocol-lock commit SHA:

Recorded by Git as the commit that introduces this final lock document and
reported in the task final response.
