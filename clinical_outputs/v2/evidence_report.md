# Screening DNN v2 — evidence report

A cross-sectional research classifier, not a clinically validated diagnosis system.

## Protocol

Exact duplicates removed; identical five-feature profiles kept in a single partition. Train / early-stop / calibrate / choose cutoff / test are disjoint. No SMOTE. NHANES used only after model, calibration and cutoff are frozen.

## Measured results

|                   |   Local test |   NHANES external (unweighted) |
|:------------------|-------------:|-------------------------------:|
| rows              | 7625         |                   3252         |
| positive_rate     |    0.107803  |                      0.187884  |
| threshold         |    0.0628417 |                      0.0628417 |
| sensitivity       |    0.922141  |                      0.9509    |
| specificity       |    0.508305  |                      0.358576  |
| precision         |    0.184743  |                      0.255385  |
| npv               |    0.981829  |                      0.969294  |
| accuracy          |    0.552918  |                      0.469865  |
| f1                |    0.307817  |                      0.402633  |
| roc_auc           |    0.816884  |                      0.779796  |
| average_precision |    0.35674   |                      0.419571  |
| brier             |    0.0818338 |                      0.12973   |
| tn                | 3458         |                    947         |
| fp                | 3345         |                   1694         |
| fn                |   64         |                     30         |
| tp                |  758         |                    581         |

## Limitations

- Dataset provenance and individual patient identifiers are unavailable; feature grouping is not a verified patient-level split.
- Local diabetes labels have undocumented ascertainment. This is cross-sectional classification, not prediction of future diabetes.
- NHANES reference is self-reported diabetes OR a single elevated HbA1c/fasting glucose; it is not confirmed clinical diagnosis.
- NHANES includes known diabetes; performance is not validated specifically for undiagnosed screening.
- NHANES smoking means at least 100 lifetime cigarettes; local smoking definitions are not documented. Harmonization is approximate.
- Local pregnancy status is unavailable; known pregnancies excluded externally, unknown pregnancy status may remain.
- External complete-case selection can introduce bias. Weighted metrics are point estimates only; no survey-design confidence intervals or population-validity claim.
- Calibration and threshold fitted locally may not transport to NHANES or a hospital. No prospective, clinical-outcome or regulatory validation.
- One fixed-seed training run, not a repeated-seed or multi-hospital study.

## Source files

- [DEMO_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm) — SHA256 `ca4374a158b493b8b0163e1388da21d57a18d1b9cecff2aa4e2fa2bec494fe23`
- [DIQ_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DIQ_L.htm) — SHA256 `9535a023673ae869afae19d842d8679e06f6a464606ac15900686b41ef05090f`
- [GHB_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.htm) — SHA256 `67aee0353160e2392dc0a33bece99b90764a630c76d82415ac4639105ad9dd03`
- [GLU_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GLU_L.htm) — SHA256 `70f773ab37e3d8485341f32137e280e05b9279e5fe9b6f0362e4052d8fe9a201`
- [BMX_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.htm) — SHA256 `44440c416d9ad709e8b1708a5975378ab4d5b18edc39eb5015c2ae7186500170`
- [BPQ_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPQ_L.htm) — SHA256 `0acde3af0526c375942fe0735e3bf69a0422bcbc606f98a478209d2f47baa4b5`
- [SMQ_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/SMQ_L.htm) — SHA256 `abbefebfa5585b1e19c079fa14be20c113a4649318070197214b258d05b93c50`