# SIPaKMeD Outcome-Blind Technical Preflight Report

Date: 2026-08-13

Protocol lock commit: `9c3d37df3a9bcf11353b5f40ed04bf98dc806d06`

Completed preflight commit: recorded in Git history for this branch.

## Scope

This is a data-only, outcome-blind technical preflight for possible future SIPaKMeD external morphometry replication. No model checkpoint was loaded, no model inference was run, no model output was inspected, and no prediction-based metric was calculated.

## Official Source

Official dataset page:

`https://www.cs.uoi.gr/~marina/sipakmed.html`

Official citation requested by the dataset page:

Marina E. Plissiti, Panagiotis Dimitrakopoulos, Giorgos Sfikas, Christophoros Nikou, Olga Krikoni, Antonia Charchanti, "SIPAKMED: A new dataset for feature and image based classification of normal and pathological cervical cells in Pap smear images", IEEE International Conference on Image Processing (ICIP), Athens, Greece, 7-10 October 2018.

Permitted use / license statement observed on the official page:

The page states that the database is publicly available and may be used for experimental purposes with a request to cite the SIPaKMeD ICIP 2018 paper. No broader open-source license text was found on the official page during this preflight.

## Archive Integrity

All five official image archives were present under `data/raw/sipakmed/`, matched the official HEAD byte counts recorded from the official source, were listable with `tar -tf`, and were extracted locally under `data/raw/sipakmed/`. Raw and extracted dataset files remain gitignored.

| Archive | Bytes | SHA256 |
| --- | ---: | --- |
| `im_Dyskeratotic.7z` | 1182980549 | `a8768bc03c9e814063ff946b7ef7548067e16b760168267d2b12e95a66955cce` |
| `im_Koilocytotic.7z` | 1290478065 | `653578e878e3cc63a8510cd7ec26d918637d9881e913e29a5bb205d0d9cfb70d` |
| `im_Metaplastic.7z` | 1396480123 | `97f1badc5f606f127d2af0d18a008ebd41a2b60f44785cb332a6eaa33004d2af` |
| `im_Parabasal.7z` | 548667699 | `13e9c29f016a5b83731e038cfe518671a7815df28836dee706acba6ecd2369a8` |
| `im_Superficial-Intermediate.7z` | 762939737 | `3b73e406df14d60b1bd9c8ac3b01b594e70bb51ba32fb080141fc6a7f489a599` |

## Complete Inventory

| Class | Parent cluster BMP | Cropped isolated BMP | Cropped nucleus contours | Cropped cytoplasm contours |
| --- | ---: | ---: | ---: | ---: |
| Dyskeratotic | 223 | 813 | 813 | 813 |
| Koilocytotic | 238 | 825 | 825 | 825 |
| Metaplastic | 271 | 793 | 793 | 793 |
| Parabasal | 108 | 787 | 787 | 787 |
| Superficial-Intermediate | 126 | 831 | 831 | 831 |
| **Total** | **966** | **4049** | **4049** | **4049** |

Total cropped annotation files: `8098`.

Image format:

- Cropped images: BMP
- Channels: RGB for all 4049 isolated-cell images
- Dtype after load: uint8 for all 4049 isolated-cell images
- Native width range: 62 to 531 pixels
- Native height range: 48 to 553 pixels
- Unique native dimensions: 3825

Duplicates and mappings:

- Duplicate sample ids: none
- Duplicate cropped image content hashes: none
- Duplicate cropped contour content hashes: none
- Missing image-to-contour mappings: none
- Corrupt/unreadable images: none detected

The deterministic sample id is:

`<diagnostic_class>__<parent_cluster_id>_<cell_id>`

The deterministic parent source id is:

`<diagnostic_class>__<parent_cluster_id>`

Parent grouping is recoverable for all samples: `966` unique parent-source ids, with 1 to 26 isolated cells per parent source.

## Contour Format And Coordinates

Direct inspection of official cropped contour files established:

- Nucleus contour format: plain text `.dat`, one comma-separated `x,y` floating-point coordinate pair per line.
- Cytoplasm contour format: plain text `.dat`, one comma-separated `x,y` floating-point coordinate pair per line.
- Coordinate convention: native image coordinate space, origin at the top-left.
- x/y order: first coordinate is horizontal `x`/column, second coordinate is vertical `y`/row.
- Indexing convention: continuous pixel coordinates bounded by `0 <= x < width` and `0 <= y < height`; these are not integer class-label indices.
- Polygon closure: contours do not need to repeat the first point; rasterization closes the polygon deterministically.

Evidence for x/y order:

- `8070 / 8098` contours were in bounds under native `x,y` interpretation.
- Only `4382 / 8098` would be in bounds if coordinates were swapped.
- Cropped contour examples align with cropped image dimensions under `x,y` interpretation.

## Cytoplasm Semantics

Verified cytoplasm semantics:

`whole-cell outer boundary including nucleus`

The cytoplasm contour files provide a single outer cell/cytoplasm boundary polygon. When rasterized, this polygon contains the nucleus for technically valid samples. Therefore the canonical SIPaKMeD target is constructed as:

- nucleus mask = rasterized nucleus contour
- whole-cell mask = rasterized cytoplasm contour
- cytoplasm-only mask = `whole_cell_mask AND NOT nucleus_mask`

This matches the existing project target convention:

- `0 = background/other`
- `1 = cytoplasm-only`
- `2 = nucleus`

## Technical Eligibility

Technically eligible isolated-cell samples: `4000`

Technically excluded isolated-cell samples: `49`

Exclusion reasons are data/annotation-only:

| Reason | Count |
| --- | ---: |
| `nucleus_contour_out_of_bounds` | 26 |
| `empty_nucleus_mask` | 25 |
| `undefined_nucleus_circularity` | 25 |
| `nucleus_pixels_outside_cell_mask` | 22 |
| `cytoplasm_contour_out_of_bounds` | 2 |

Counts may overlap because one sample can have more than one technical exclusion reason.

For all technically eligible samples:

- nucleus area is defined
- nucleus perimeter is defined
- nucleus circularity is defined
- cytoplasm-only area is nonzero
- N/C ratio is mathematically computable
- nucleus and cytoplasm-only masks are pixel-exclusive by construction
- nucleus is geometrically compatible with the cell mask

N/C is not computable for all raw samples because 49 samples fail technical annotation validity checks. These failures are explicitly listed in `results/sipakmed_preflight/sipakmed_eligibility_manifest.csv`.

## Outputs

Generated outcome-blind data-only outputs:

- `src/data_prep/load_sipakmed.py`
- `experiments/run_sipakmed_preflight.py`
- `tests/test_sipakmed_loader.py`
- `results/sipakmed_preflight/sipakmed_eligibility_manifest.csv`
- `results/sipakmed_preflight/gt_alignment_qc.png`
- `results/sipakmed_preflight/preflight_summary.json`

The QC figure contains RGB images with GT nucleus and GT whole-cell/cytoplasm overlays only. It contains no model predictions.

## Existing Herlev Preprocessing Compatibility

The existing Herlev preprocessing is technically compatible with the eligible SIPaKMeD isolated-cell images:

- RGB input can be preserved.
- Pixel values are uint8 and can be normalized from `[0, 255]` to `[0, 1]`.
- Aspect-ratio-preserving resize and center padding to `128 x 128` can be applied.
- Native-geometry GT masks are available for future restoration/evaluation.

No SIPaKMeD image was passed through a model in this preflight.

## Verdict

`GO`

Rationale:

The full official data are available and extracted; image/annotation mapping is deterministic; nucleus and cytoplasm contours are present and directly parsed; coordinate convention is resolved; cytoplasm semantics are resolved as whole-cell outer boundary; canonical 3-class GT construction is reproducible; N/C is computable for the locked technically eligible subset; parent-source grouping is recoverable. The 49 exclusions are deterministic technical annotation-validity exclusions and are recorded before any model exposure.

The dataset is ready for future external protocol freezing. Do not run inference until that protocol is explicitly frozen.

## Outcome-Blind Compliance

- Was any model checkpoint loaded? NO
- Was any inference run? NO
- Were any prediction metrics inspected? NO
- Was training or fine-tuning run? NO
- Were existing Herlev/closing/matched-null scientific results changed? NO
- Were raw SIPaKMeD files committed? NO
