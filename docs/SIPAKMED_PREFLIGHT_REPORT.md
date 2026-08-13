# SIPaKMeD Outcome-Blind Technical Preflight Report

Date: 2026-08-13

Protocol lock commit: `9c3d37df3a9bcf11353b5f40ed04bf98dc806d06`

## Scope

This report is an outcome-blind technical preflight for possible future SIPaKMeD external morphometry replication. No model checkpoint was loaded, no model prediction was generated, no inference was run, and no prediction-vs-ground-truth metric was calculated.

## Official Source

Official dataset page:

`https://www.cs.uoi.gr/~marina/sipakmed.html`

Official citation requested by the dataset page:

Marina E. Plissiti, Panagiotis Dimitrakopoulos, Giorgos Sfikas, Christophoros Nikou, Olga Krikoni, Antonia Charchanti, "SIPAKMED: A new dataset for feature and image based classification of normal and pathological cervical cells in Pap smear images", IEEE International Conference on Image Processing (ICIP), Athens, Greece, 7-10 October 2018.

Permitted use / license statement observed on the official page:

The page states that the database is publicly available and may be used for experimental purposes with a request to cite the SIPaKMeD ICIP 2018 paper. No broader open-source license text was found on the official page during this preflight.

## Official Download Routes

All links below were found on the official page. Mirrors were not used.

| Archive | Official route | HEAD status | Reported size bytes |
| --- | --- | ---: | ---: |
| `im_Superficial-Intermediate.7z` | `https://www.cs.uoi.gr/~marina/SIPAKMED/im_Superficial-Intermediate.7z` | 200 | 762939737 |
| `im_Parabasal.7z` | `https://www.cs.uoi.gr/~marina/SIPAKMED/im_Parabasal.7z` | 200 | 548667699 |
| `im_Koilocytotic.7z` | `https://www.cs.uoi.gr/~marina/SIPAKMED/im_Koilocytotic.7z` | 200 | 1290478065 |
| `im_Metaplastic.7z` | `https://www.cs.uoi.gr/~marina/SIPAKMED/im_Metaplastic.7z` | 200 | 1396480123 |
| `im_Dyskeratotic.7z` | `https://www.cs.uoi.gr/~marina/SIPAKMED/im_Dyskeratotic.7z` | 200 | 1182980549 |
| `Features_CELL.7z` | `https://www.cs.uoi.gr/~marina/SIPAKMED/Features_CELL.7z` | 200 | 608820 |
| `Description_of_Features.pdf` | `https://www.cs.uoi.gr/~marina/SIPAKMED/Description_of_Features.pdf` | 200 | 478395 |

The five image archives total approximately 5.18 GB before extraction.

## Local Download Status

Downloaded from the official source into ignored raw storage:

- `data/raw/sipakmed/Description_of_Features.pdf`
- `data/raw/sipakmed/Features_CELL.7z`

Checksums:

- `Description_of_Features.pdf`: SHA256 `4006bbee2ca0ed18cf12cd9ec3c1d7f2d203e716328e995f88e3365c97cf27b9`
- `Features_CELL.7z`: SHA256 `23cebcab368e337e135d7c1bb39bc8a46bdfd19fbe20c0c4d74be4e003ee5688`

Automatic download of `im_Superficial-Intermediate.7z` from the official host did not complete within a 40-minute command timeout. The incomplete local partial file was removed to avoid future confusion. The remaining official image archives were not downloaded in this run.

Expected manual download destination:

`data/raw/sipakmed/`

Manual download instructions:

1. Open `https://www.cs.uoi.gr/~marina/sipakmed.html`.
2. Download the five official image archives listed above into `data/raw/sipakmed/`.
3. Do not use Kaggle or other mirrors for this preflight.
4. Keep the raw files uncommitted.
5. After download, rerun the preflight inventory on the official archives only.

## Official Dataset Description

The official page states that SIPaKMeD consists of 4049 isolated-cell images manually cropped from 966 cluster-cell images of Pap smear slides. The official page lists five categories: Superficial-Intermediate, Parabasal, Koilocytotic, Dyskeratotic, and Metaplastic.

The official `Description_of_Features.pdf` states that boundaries of the cytoplasm and nucleus regions were manually defined by expert observers, and that contour coordinates are provided for both cluster images and isolated-cell images in the class image directories and their `CROPPED` subdirectories.

## File-Level Inventory

Full file-level image and contour inventory was not performed because the official image archives were not fully available locally.

Source-level verified counts:

- Isolated-cell images: 4049, from official page.
- Parent cluster images: 966, from official page.
- Category count: 5, from official page.
- Feature tables: 10 `.dat` tables extracted from `Features_CELL.7z`, one cytoplasm table and one nuclei table per category.

Local extracted feature files:

- `DYSKERATOTIC_CYTOPLASM_FEAT.dat`
- `DYSKERATOTIC_NUCLEI_FEAT.dat`
- `KOILOCYTOTIC_CYTOPLASM_FEAT.dat`
- `KOILOCYTOTIC_NUCLEI_FEAT.dat`
- `METAPLASTIC_CYTOPLASM_FEAT.dat`
- `METAPLASTIC_NUCLEI_FEAT.dat`
- `PARABASAL_CYTOPLASM_FEAT.dat`
- `PARABASAL_NUCLEI_FEAT.dat`
- `SUP_INT_CYTOPLASM_FEAT.dat`
- `SUP_INT_NUCLEI_FEAT.dat`

These feature files contain derived intensity, texture, and shape features; they are not a substitute for contour files.

## Annotation Semantics

Nucleus GT status:

Official documentation states that nucleus contour coordinates are provided.

Cytoplasm GT status:

Official documentation states that cytoplasm contour coordinates are provided.

Meaning of cytoplasm contour:

Not fully established from local contour files. The official PDF describes "the area of the cytoplasm and the nucleus" and separately names cytoplasm and nucleus feature tables. This supports the presence of a cytoplasm region annotation, but the whole-cell-including-nucleus vs cytoplasm-only distinction must still be verified directly from the contour coordinate files after the official image archives are available.

N/C computable:

Conditionally yes, if the contour files align with the official description and the cytoplasm contour semantics can be resolved deterministically. It is not yet locally verified.

Image-annotation mapping:

Not yet locally verified. The feature tables use image number and cell number fields, and the official PDF says contour coordinates are provided in the class image directories and `CROPPED` subdirectories. Full deterministic mapping requires the image archives.

Parent-source grouping:

Potentially recoverable through cluster image id and cell id, but not yet locally verified from the archive structure. Future inference should use parent-source-aware resampling if parent source ids are confirmed.

## Existing Herlev Preprocessing Compatibility

The existing preprocessing in `src/segmentation/train_unet.py` is statically compatible in principle with RGB isolated-cell images:

- RGB input is preserved.
- Pixel values are normalized from uint8 `[0, 255]` to float `[0, 1]`.
- Aspect ratio is preserved by isotropic resize.
- Center padding is applied to `128 x 128`.
- Masks are resized with nearest-neighbor interpolation when target masks are built.

No SIPaKMeD image was passed through model preprocessing in this preflight.

## Technical Eligibility Rules For Future Full Preflight

Future inclusion/exclusion must be based only on technical data validity. Proposed exclusion reasons:

- unreadable image
- missing corresponding nucleus contour
- missing corresponding cytoplasm or whole-cell contour
- contour parse failure
- self-intersecting or irrecoverably invalid polygon
- contour coordinates outside the image with no documented correction
- empty nucleus mask after rasterization
- empty cytoplasm-only mask after rasterization
- unresolvable image/cell id mismatch
- unresolvable coordinate indexing or x/y orientation ambiguity

Forbidden exclusion reasons:

- low predicted Dice
- high morphometric error
- diagnosis or category performing poorly
- difficult-looking morphology
- future model failure
- post-processing failure

## Verdict

`CONDITIONAL_GO`

Rationale:

The official source is confirmed and official documentation states that both nucleus and cytoplasm contour coordinates are provided for the cluster and isolated-cell image directories. This makes SIPaKMeD technically plausible for a future nucleus-plus-cytoplasm morphometry preflight. However, the full official image archives were not available locally after the automatic download attempt timed out, so contour file format, contour semantics, image-annotation alignment, parent-source grouping, malformed-file rates, native geometry distributions, and deterministic loader behavior remain unverified.

The next step must be a full official-archive download followed by a data-only archive/file/contour inventory. No model inference should be run before that inventory passes.

## Outcome-Blind Compliance

- Was any model checkpoint loaded? NO
- Was any inference run? NO
- Were any prediction metrics seen? NO
- Was training or fine-tuning run? NO
- Were existing Herlev/closing/matched-null scientific results changed? NO
- Were raw SIPaKMeD files committed? NO
