# SIPaKMeD Outcome-Blind Technical Preflight Protocol

## Purpose

This protocol locks an outcome-blind technical preflight for assessing whether SIPaKMeD can be converted into a technically defensible, reproducible, nucleus-plus-cytoplasm external morphometry dataset for the existing frozen Herlev segmentation model.

This phase is data and annotation feasibility only. It is not an experiment, not external validation, and not a model-performance analysis.

## Non-Negotiable Outcome-Blind Rules

1. No checkpoint may be loaded.
2. No model prediction may be generated.
3. No Dice, IoU, precision, recall, segmentation error, circularity error, N/C error, area error, perimeter error, or any other prediction-vs-ground-truth result may be calculated.
4. No inclusion or exclusion rule may depend on a model prediction.
5. No image may be selected because it looks easy or difficult.
6. No intervention or post-processing result may be examined.
7. No training, fine-tuning, checkpoint selection, lambda selection, threshold tuning, or domain adaptation may be performed.
8. Existing Herlev results, closing-correction results, matched-null results, checkpoints, splits, preregistrations, and final decision locks must not be changed.

If a checkpoint is accidentally loaded or a model prediction is accidentally generated, the preflight must stop immediately and the exposure must be documented.

## Allowed Inspection Scope

The preflight may inspect only dataset-side technical properties needed to determine whether a future frozen-model external replication could be defined reproducibly:

- file existence
- file names
- archive names, sizes, and checksums when feasible
- image dimensions
- image formats
- RGB/grayscale/channel structure
- dtype and numeric range where technically relevant
- annotation structure
- contour validity
- number of images and annotations
- parent-image identifiers
- image/annotation alignment
- malformed or missing files
- duplicate filenames and duplicate content hashes
- number of contour points
- whether nucleus and cytoplasm contours can be rasterized
- whether nucleus lies within cytoplasm or whole-cell masks
- whether cytoplasm-only masks can be constructed
- whether technical pixel and geometry properties support existing preprocessing

Dataset-only morphometric values may be calculated only for technical validity checks and must be labeled:

`DATA CHARACTERIZATION ONLY - NO MODEL OUTCOMES`

These values must not be used to select convenient cells, tune preprocessing, define performance thresholds, or make scientific claims about model behavior.

## Annotation Semantics To Establish Before Use

The preflight must document evidence for:

- what the nucleus annotation represents
- what the cytoplasm annotation represents
- whether the cytoplasm contour is a whole-cell contour including the nucleus or a cytoplasm-only contour
- whether every usable cell has exactly one nucleus contour
- whether every usable cell has exactly one cytoplasm or whole-cell contour
- whether multiple nuclei, self-intersections, out-of-bounds contours, empty masks, zero cytoplasm-only area, coordinate-index ambiguity, or x/y-order ambiguity occur

The canonical target definition for possible future work is:

- `0 = background/other`
- `1 = cytoplasm-only`
- `2 = nucleus`

If SIPaKMeD provides a whole-cell contour, cytoplasm-only must be constructed as:

`whole_cell_mask AND NOT nucleus_mask`

If SIPaKMeD provides a cytoplasm-only contour, that mask may be used directly only after annotation semantics are documented.

## Deterministic Eligibility Rules

Future eligibility decisions may depend only on technical annotation validity. Acceptable exclusion reasons include:

- unreadable image
- missing corresponding nucleus annotation
- missing corresponding cytoplasm or whole-cell annotation
- irrecoverably invalid polygon
- empty nucleus after rasterization
- empty cytoplasm-only mask
- unresolvable coordinate mismatch
- annotation-to-image mapping failure

Not acceptable exclusion reasons include:

- low predicted Dice
- high morphometric error
- difficult-looking morphology
- diagnosis or category performing poorly
- model failure
- post-processing failure

## Future Preprocessing Boundary

This preflight may inspect the existing Herlev preprocessing code and decide whether SIPaKMeD can enter it without data-driven tuning. The preferred future route, if technically valid, is:

native SIPaKMeD isolated cell -> existing frozen Herlev preprocessing -> frozen U-Net -> restore prediction to SIPaKMeD native geometry -> morphometry

This route must not be executed in this preflight.

No CLAHE, stain normalization, sharpening, color matching, histogram matching, crop tuning, or other domain adaptation step may be introduced to make SIPaKMeD look more like Herlev.

## Outputs Planned For This Preflight

If official data are unavailable due to manual authentication, browser confirmation, anti-bot protection, or license acceptance, the preflight must stop after producing exact manual download instructions and the expected local destination:

`data/raw/sipakmed/`

If the official files are available locally and annotation structure is sufficiently clear, the preflight may produce:

- `src/data_prep/load_sipakmed.py`
- deterministic loader tests
- `results/sipakmed_preflight/sipakmed_eligibility_manifest.csv`
- `results/sipakmed_preflight/gt_alignment_qc.png`
- `results/sipakmed_preflight/preflight_summary.json`
- `docs/SIPAKMED_PREFLIGHT_REPORT.md`
- `docs/SIPAKMED_EXTERNAL_REPLICATION_DRAFT.md` only if the verdict is `GO` or `CONDITIONAL_GO`

All raw SIPaKMeD files must remain gitignored.

## Verdict Criteria

The preflight verdict must be one of:

- `GO`: official source confirmed; images and nucleus/cytoplasm or whole-cell ground truth are available; deterministic mapping exists; canonical three-class target can be built; N/C is computable; technically valid cells are sufficient; parent grouping is recoverable or manageable; existing preprocessing can be applied without tuning.
- `CONDITIONAL_GO`: the dataset appears scientifically useful, but a documented technical limitation remains, such as unavailable parent IDs or minor annotation ambiguity.
- `NO_GO`: nucleus/cytoplasm mapping is unreliable; contours cannot be aligned reproducibly; isolated-cell images are unavailable; the dataset requires subjective cell selection; or required annotation meaning cannot be established.

The verdict must not depend on future model performance.
