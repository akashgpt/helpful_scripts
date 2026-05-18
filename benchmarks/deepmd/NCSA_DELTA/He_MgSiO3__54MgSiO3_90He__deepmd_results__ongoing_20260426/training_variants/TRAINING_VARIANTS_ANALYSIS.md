# training_bench analysis

Source: `/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench`

## Completed runs

| variant | status | backend | desc | steps | s/batch | final F | late100 med F | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| variant_train_se_e2_a_TF | complete | TF | se_e2_a | 1000000/1000000 | 0.0386 | 0.36 | 0.3485 | finished training |
| variant_train_se_e2_a_TF_fit_deep2x | complete | TF | se_e2_a | 1000000/1000000 | 0.0449 | 0.351 | 0.344 | finished training |
| variant_train_se_e2_a_TF_big | complete | TF | se_e2_a | 1000000/1000000 | 0.0805 | 0.391 | 0.3815 | finished training |
| variant_train_se_e2_a_TF_both_deep2x | complete | TF | se_e2_a | 1000000/1000000 | 0.0994 | 0.459 | 0.4505 | finished training |
| variant_train_se_e2_a_TF_fit_deep10x | complete | TF | se_e2_a | 1000000/1000000 | 0.1003 | 0.375 | 0.356 | finished training |
| variant_train_se_e2_a_TF_balanced_10x | complete | TF | se_e2_a | 1000000/1000000 | 0.163 | 0.371 | 0.373 | finished training |

## Incomplete/staged runs

| variant | status | backend | desc | steps | s/batch | final F | late100 med F | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| variant_train_dpa2_PT | partial | PT | dpa2 | 1000000/1000000 | 0.1011 | 3.49 | 3.71 | has lcurve but no finished-training marker |
| variant_train_dpa2_PT_big | partial | PT | dpa2 | 1000000/1000000 | 0.0991 | 3.49 | 3.71 | has lcurve but no finished-training marker |
| variant_train_dpa2_PT_big_deep_v2 | staged | PT |  |  |  |  |  | script/config only; no training output |
| variant_train_dpa2_PT_big_strip | partial | PT | dpa2 | 1000000/1000000 | 0.1028 | 3.49 | 3.71 | has lcurve but no finished-training marker |
| variant_train_dpa2_PT_big_strip_v2 | failed | PT | dpa2 | 499900/1000000 |  | 3.76 | 3.68 | filesystem input/output error while reading data |
| variant_train_dpa2_PT_big_v2 | failed | PT | dpa2 | 556100/1000000 |  | 2.85 | 3.76 | PyTorch dataloader bus error / shm pressure |
| variant_train_dpa2_PT_v2 | failed | PT | dpa2 | 972800/1000000 |  | 3.29 | 3.555 | PyTorch dataloader bus error / shm pressure |
| variant_train_se_e2_a_PT | partial | PT | se_e2_a | 1000000/1000000 | 0.1024 | 0.598 | 0.352 | has lcurve but no finished-training marker |
| variant_train_se_e2_a_TF_balanced_2x | partial | TF | se_e2_a | 297900/1000000 |  | 0.481 | 0.3585 | has lcurve but did not reach configured steps |
| variant_train_se_e2_a_TF_balanced_5x | partial | TF | se_e2_a | 164900/1000000 |  | 0.349 | 0.37 | has lcurve but did not reach configured steps |
| variant_train_se_e2_a_TF_big2x | partial | TF | se_e2_a | 341000/1000000 |  | 0.338 | 0.369 | has lcurve but did not reach configured steps |
| variant_train_se_e2_a_TF_big5x | partial | TF | se_e2_a | 262800/1000000 |  | 0.407 | 0.3605 | has lcurve but did not reach configured steps |
| variant_train_se_e2_a_TF_big_deep | staged | TF |  |  |  |  |  | script/config only; no training output |
| variant_train_se_e2_a_TF_both_deep10x | failed | TF | se_e2_a | 563200/1000000 |  | 1.28 | 0.8985 | TensorFlow/DeepMD out-of-memory |
| variant_train_se_e2_a_TF_desc_deep1 | staged | TF |  |  |  |  |  | script/config only; no training output |
| variant_train_se_e2_a_TF_desc_deep10x | failed | TF | se_e2_a | 301500/1000000 |  | 3.07 | 3.72 | TensorFlow/DeepMD out-of-memory |
| variant_train_se_e2_a_TF_desc_deep2x | failed | TF | se_e2_a | 527900/1000000 |  | 0.262 | 0.3665 | NaN detected during training |
| variant_train_se_e2_a_TF_fit_deep5x | partial | TF | se_e2_a | 234100/1000000 |  | 0.357 | 0.364 | has lcurve but did not reach configured steps |
