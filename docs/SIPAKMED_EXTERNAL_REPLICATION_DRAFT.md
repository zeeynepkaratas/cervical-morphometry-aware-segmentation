# SIPaKMeD External Morphometry Replication Draft

Status: Draft only. Do not execute until the SIPaKMeD technical preflight is completed from fully downloaded official archives.

## Primary Scientific Question

Does the relationship between segmentation overlap and downstream morphometric fidelity observed in Herlev replicate in SIPaKMeD?

## Future Frozen Model

Use only the existing Herlev baseline checkpoint family.

Do not train, fine-tune, select checkpoints on SIPaKMeD, tune thresholds, tune post-processing, or perform domain adaptation on SIPaKMeD.

## Required Preconditions

This draft may be executed only after:

- official SIPaKMeD image archives are fully downloaded from the official source
- nucleus contours are confirmed
- cytoplasm or whole-cell contours are confirmed
- cytoplasm contour semantics are resolved
- image-to-annotation mapping is deterministic
- parent-source grouping is recovered or explicitly documented as unavailable
- a native-geometry SIPaKMeD loader passes deterministic unit tests
- a GT-only QC panel confirms annotation alignment
- the eligibility manifest is locked using data-only rules

## Proposed Future Pipeline

native SIPaKMeD isolated cell -> existing frozen Herlev preprocessing -> frozen U-Net -> restore prediction to SIPaKMeD native geometry -> morphometry

No part of this pipeline is executed by the preflight.

## Candidate Future Metrics

Segmentation:

- nucleus Dice
- cytoplasm Dice or foreground Dice, depending on confirmed annotation semantics

Morphometry:

- nucleus area absolute error
- nucleus perimeter absolute error
- circularity absolute error
- N/C absolute error

Association analysis:

- Spearman correlation between nucleus Dice and circularity absolute error
- Spearman correlation between nucleus Dice and nucleus area absolute error
- Spearman correlation between nucleus Dice and perimeter absolute error
- association between foreground/cytoplasm overlap and N/C absolute error

## Estimation Principles

Prefer effect sizes and confidence intervals over binary significant/not-significant declarations.

Do not freeze arbitrary success thresholds unless they are justified before any SIPaKMeD model outcome is inspected.

If multiple isolated cells originate from the same parent source image, retain the cell as the measurement unit but use parent-source-cluster-aware resampling for confidence intervals.

For multiple baseline seeds, aggregate seed predictions and effects according to the existing manuscript's no-pseudoreplication principle.

## Prohibited During Future Replication

- training on SIPaKMeD
- fine-tuning on SIPaKMeD
- checkpoint selection using SIPaKMeD
- post-processing selection using SIPaKMeD outcomes
- excluding samples based on model failure or performance
- changing Herlev preprocessing to make SIPaKMeD visually closer to Herlev
