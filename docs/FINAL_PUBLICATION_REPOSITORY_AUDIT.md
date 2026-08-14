# Final Publication Repository Audit

Audit date: 2026-08-14

Repository: `zeeynepkaratas/cervical-morphometry-aware-segmentation`

Verdict: `PUBLICATION_REPOSITORY_READY`

This audit is repository/provenance/documentation cleanup only. No new
scientific experiment was run, no model was trained or fine-tuned, no
SIPaKMeD eligibility rule was changed, and no locked protocol was modified.

## 1. Final Main Commit

The final publication-ready main commit is the main-branch commit containing
this audit document. The exact SHA is recorded in the task final response after
the normal merge and push to `main`.

## 2. SIPaKMeD Lock And Result Commits

| Milestone | SHA |
| --- | --- |
| SIPaKMeD preflight GO | `ab66b33314d23fda686e236150c354a915f21a7f` |
| SIPaKMeD final protocol lock | `b0cb7f6d727503861a36b0d754f18c5d999cffe8` |
| SIPaKMeD external replication result | `c3fc4e0fc72a92ce50649d286c0814b224b2ca11` |

Ancestry audit:

- `ab66b33314d23fda686e236150c354a915f21a7f -> b0cb7f6d727503861a36b0d754f18c5d999cffe8`: PASS
- `b0cb7f6d727503861a36b0d754f18c5d999cffe8 -> c3fc4e0fc72a92ce50649d286c0814b224b2ca11`: PASS

## 3. Lock Tag

Annotated lock tag:

`sipakmed-external-lock-20260813`

Verified target:

`b0cb7f6d727503861a36b0d754f18c5d999cffe8`

Status: PASS

## 4. Publication-Ready Tag

Publication-ready tag:

`manuscript-ready-sipakmed-20260814`

The tag points to the final publication-ready `main` commit created after this
audit passes. The exact target SHA is recorded in the task final response.

## 5. Result Consistency Audit

Authoritative artefacts:

- `results/sipakmed_external_replication/aggregate_summary.json`
- `results/sipakmed_external_replication/bootstrap_results.csv`
- `docs/SIPAKMED_EXTERNAL_REPLICATION_REPORT.md`

Headline values agree between the report, summary JSON, and bootstrap CSV after
normal display rounding.

| Outcome | Frozen value |
| --- | ---: |
| n evaluated | `4000 / 4000` |
| Primary nucleus Dice | `0.664344 [0.649495, 0.679036]` |
| Primary circularity AE | `0.185073 [0.177883, 0.192633]` |
| Dice-circularity AE Spearman rho | `-0.505682 [-0.538004, -0.471973]` |
| Area AE | `1717.59975 [1555.16709, 1893.22372]` |
| Perimeter AE | `79.582070 [73.370307, 86.295514]` |
| N/C AE | `0.153275 [0.138359, 0.170207]` |
| Closing benefit | `0.054938 [0.051785, 0.058029]` |
| Matched-null benefit | `0.052296 [0.049260, 0.055286]` |
| Closing-specific residual | `0.002306 [0.002046, 0.002585]` |

Consistency verdict: PASS

Note: paired residuals and paired benefits are estimated from cell-level paired
effects with seed-family aggregation. They should not be expected to equal
simple subtraction of separately rounded headline means.

## 6. Checkpoint Hashes

| Seed | SHA256 |
| ---: | --- |
| 20260803 | `8c52642452c674bc4b929a7b643c9f69f0fc600d2f7b5eafdf733287e98eed51` |
| 20260804 | `d8bf421f71b0ba899a9d70700f6bdb1155110fd91af8617daff97325aed2b6af` |
| 20260805 | `a8e1563e5dc86f67ed5b393a8f1fc78a96165b727cedd766aa8c1131bc5fdc26` |

Status: PASS

## 7. Dataset Manifest Hashes

| Artefact | SHA256 |
| --- | --- |
| `results/sipakmed_preflight/sipakmed_eligibility_manifest.csv` | `e86e99441866fb03d457e3867032ecee4e6792285c6647d93e35089b923de1cd` |
| `results/sipakmed_preflight/preflight_summary.json` | `c359aaaa50b4ed06c8ea086f8e3d5ac8ba6a856d5c0b7e3ec2cc2df9165f9844` |

## 8. Major Result Artefact Hashes

| Artefact | SHA256 |
| --- | --- |
| `results/sipakmed_external_replication/aggregate_summary.json` | `3d6c2b52f65baa741b1c9c1afa1242dde9554efd17eb56609a671cdd7fa4546b` |
| `results/sipakmed_external_replication/bootstrap_results.csv` | `b1792a477fcaafd6782aca91a18f32b6a57c6d50ac2bc5c100fdb8e40e1510d4` |
| `results/sipakmed_external_replication/cell_median_raw_metrics.csv` | `bf62e2eb1fcd85f76d207d17a60b50e755039b8a850692c752bfa6dd307af25c` |
| `results/sipakmed_external_replication/cell_median_specificity.csv` | `1ea852469a694f95aacc8118b2e6bf2cb4cee76e64229951d3f6bab07adfd72f` |
| `results/sipakmed_external_replication/cell_seed_processing_metrics.csv` | `760173dba8b72f4d45ed8cdf4ec0b11f4aac9f9663392150f70ac0f84e6ae26d` |
| `results/sipakmed_external_replication/cell_seed_raw_metrics.csv` | `f8f205e349e9fb4b8373a88aeb1967620f1bc4dd239938408b5fcf1e4feed562` |
| `results/sipakmed_external_replication/cell_seed_specificity.csv` | `af8f8e1ab440df6e301e136225549f61c28a531ff6c0c6868db9931a0b585f09` |
| `results/sipakmed_external_replication/runtime_environment.json` | `91e3e93e56b5ac80eb1c461c25a7b4765ad1e1680a7a1d3734daec9b4077df15` |
| `results/sipakmed_external_replication/checkpoint_hashes.json` | `0e7294adb8b7955c44a062d0f556f6d6110c8794d27c7603e5a2c3a4040bf862` |
| `results/sipakmed_external_replication/provenance_hashes.json` | `5ba9c83d9fd87494829bca0e8b0b3a026b3f32a4472c96386ce7c1fccc906590` |

## 9. Relevant Implementation Hashes

| Artefact | SHA256 |
| --- | --- |
| `docs/SIPAKMED_EXTERNAL_REPLICATION_FINAL_LOCK.md` | `e9824a64a3ed586e3bcb9e5e85f3a644a2bd1d73e1c8fad090bd7d3f6907eb45` |
| `docs/SIPAKMED_EXTERNAL_REPLICATION_REPORT.md` | `688da78c702453074f988993d4668ac9674f26e84d83cf29d378fa002c15520e` |
| `experiments/run_sipakmed_external_replication.py` | `403c0a871670d6f09e80276c9fb9dab4133b6534461387c095024cd13be45335` |
| `src/data_prep/load_sipakmed.py` | `a20d049aff5f95ce93fe737b5e3649c48a7c5f33d4b7e44bd81941386f5f857a` |

## 10. Tests And Syntax Checks

Full test suite:

`56 passed`

Publication-critical script syntax checks:

- `experiments/run_sipakmed_external_replication.py`: PASS

## 11. Branches Merged

Required completed branches for publication consolidation:

- `codex/sipakmed-preflight`
- `codex/sipakmed-external-replication`

The external replication branch descends from the preflight branch and contains
the final lock and result artefacts. It is merged into `main` by normal Git
merge. Feature branches are preserved for provenance.

## 12. Raw-Data Tracking Status

Raw SIPaKMeD data exists locally under `data/raw/sipakmed/`, including official
archives and extracted files, but Git tracking audit found:

- tracked `data/raw/sipakmed` files: `0`
- tracked `.7z` archives: `0`
- tracked raw `.bmp` files: `0`

Status: PASS

## 13. Temporary Files Removed Or Ignored

Temporary local runtime logs from the SIPaKMeD run were removed after successful
completion because final CSV/JSON/figure/provenance artefacts exist.

`.gitignore` now explicitly covers:

- raw data and archives,
- Python caches and virtual environments,
- runtime logs,
- result caches,
- resume partial JSONL artefacts.

Final scientific result CSV/JSON files are not ignored.

## 14. Stale Documentation Corrected

Publication-facing current documents updated:

- `README.md`
- `docs/EXPERIMENT_STATUS.md`
- `docs/SOURCE_PROVENANCE.md`

Historical preregistration, protocol, and lock documents were not rewritten.
Superseded or chronological documents are preserved as provenance.

## 15. Unresolved Issues

No repository-blocking issue remains for the publication-ready state.

Known scope limitations to preserve in manuscript wording:

- This is not clinical validation.
- Cx22 is historical/supportive context, not untouched confirmatory validation.
- Closing should not be presented as a uniquely validated morphology correction;
  the matched-null results show most apparent circularity improvement is
  reproduced by GT-blind geometry-matched perturbation.

Final verdict: `PUBLICATION_REPOSITORY_READY`
