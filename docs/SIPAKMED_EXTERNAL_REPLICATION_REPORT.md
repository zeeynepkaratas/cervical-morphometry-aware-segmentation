# SIPaKMeD External Replication Report

Protocol lock commit: `b0cb7f6d727503861a36b0d754f18c5d999cffe8`
Run branch: `codex/sipakmed-external-replication`
Locked eligible samples evaluated: `4000` / `4000`
All 4000 locked eligible samples evaluated: `True`

This report follows the frozen outcome-blind protocol. No training, fine-tuning, checkpoint selection, preprocessing tuning, sample reselection, or post-processing parameter sweep was performed.

## 1. Primary External Replication

| Outcome | Estimate [parent-source cluster 95% CI] |
| --- | ---: |
| Mean nucleus Dice | `0.664344 [0.649495, 0.679036]` |
| Mean nucleus circularity AE | `0.185073 [0.177883, 0.192633]` |
| Spearman: nucleus Dice vs circularity AE | `-0.505682 [-0.538004, -0.471973]` |

## 2. Secondary Morphometric Analyses

| Outcome | Estimate [parent-source cluster 95% CI] |
| --- | ---: |
| Mean nucleus area AE | `1717.6 [1555.17, 1893.22]` |
| Mean nucleus perimeter AE | `79.5821 [73.3703, 86.2955]` |
| Mean N/C AE | `0.153275 [0.138359, 0.170207]` |
| Mean foreground Dice | `0.680539 [0.668306, 0.692455]` |
| Spearman: foreground Dice vs area AE | `-0.591968 [-0.622711, -0.560352]` |
| Spearman: foreground Dice vs perimeter AE | `-0.577711 [-0.606739, -0.547826]` |
| Spearman: foreground Dice vs N/C AE | `-0.546336 [-0.585011, -0.50565]` |
| Spearman: nucleus Dice vs area AE | `-0.730027 [-0.749445, -0.709087]` |
| Spearman: nucleus Dice vs perimeter AE | `-0.663435 [-0.687443, -0.638758]` |

## 3. External Closing-vs-Matched-Null Specificity Replication

| Outcome | Estimate [parent-source cluster 95% CI] |
| --- | ---: |
| Mean raw circularity AE | `0.185073 [0.177883, 0.192633]` |
| Mean fixed-closing circularity AE | `0.129863 [0.123755, 0.136371]` |
| Mean matched-null circularity AE | `0.132576 [0.126288, 0.139208]` |
| Mean closing benefit | `0.0549377 [0.051785, 0.0580288]` |
| Mean matched-null benefit | `0.0522959 [0.0492598, 0.0552861]` |
| Mean closing-specific residual | `0.00230618 [0.00204596, 0.00258472]` |

## Output Artifacts

- `results\sipakmed_external_replication\cell_seed_processing_metrics.csv`
- `results\sipakmed_external_replication\cell_seed_raw_metrics.csv`
- `results\sipakmed_external_replication\cell_seed_specificity.csv`
- `results\sipakmed_external_replication\cell_median_raw_metrics.csv`
- `results\sipakmed_external_replication\cell_median_specificity.csv`
- `results\sipakmed_external_replication\seed_summary.csv`
- `results\sipakmed_external_replication\bootstrap_results.csv`
- `results\sipakmed_external_replication\aggregate_summary.json`
- `results\sipakmed_external_replication\runtime_environment.json`
- `results\sipakmed_external_replication\checkpoint_hashes.json`
- `results\sipakmed_external_replication\provenance_hashes.json`
- `results\sipakmed_external_replication\figures\primary_dice_vs_circularity_ae.png`
- `results\sipakmed_external_replication\figures\closing_specificity_circularity_ae.png`
- `results\sipakmed_external_replication\figures\secondary_morphometry_mean_ae.png`

## Runtime/Data Failures

Runtime/data failures: `0`

## Outcome Discipline

- Protocol changed after seeing SIPaKMeD outcomes: NO
- Eligibility changed after seeing SIPaKMeD outcomes: NO
- Checkpoints selected based on SIPaKMeD performance: NO
- Post-processing parameters swept on SIPaKMeD: NO

## Bootstrap Records

Bootstrap rows written: `18`
Bootstrap repeats: `10000`
Bootstrap seed: `20260813`
