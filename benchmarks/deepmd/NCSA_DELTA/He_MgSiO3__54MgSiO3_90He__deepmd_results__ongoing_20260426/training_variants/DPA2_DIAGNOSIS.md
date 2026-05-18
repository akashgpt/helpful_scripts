# DPA-2 Diagnosis

Snapshot checked on 2026-05-18 from:

`/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench`

## Main Read

The DPA-2 results in this benchmark should not be interpreted as a clean failure of
the DPA-2 architecture. They are mostly evidence that the DPA-2 setup used here was not
well matched to this dense training set and was also fragile under the PyTorch data path.

The documentation/literature check strengthens this conclusion. DeePMD describes DPA-2
as a two-stage descriptor (`repinit` then `repformer`) that updates local
neighbor-based representations with message passing and attention. The DPA-2 paper's
main advantage is not merely "use a bigger descriptor from scratch"; it is the
large-atomic-model workflow: multi-task pre-training, downstream fine-tuning, and
distillation. Our benchmark DPA-2 runs were single-task, from-scratch PyTorch runs on a
MgSiOH speed-benchmark dataset, not fine-tuned/pretrained DPA-2 models on the
He/MgSiO3 target task.

## Evidence

- The original DPA-2 configs used undersized neighbor selections.
  - `variant_train_dpa2_PT` warned that the 6 A stage needed at least `324` neighbors
    but `repinit.nsel` was `120`.
  - The 4 A repformer stage needed at least `107` neighbors, while the original config
    was effectively `30` to `40` depending on the internal stage.
- The v2 DPA-2 configs only partly fixed this.
  - `repinit.nsel=360` removed the first 6 A warning.
  - `repformer.nsel=120` still produced an effective-selection warning: needed at
    least `107`, but the run reported an effective value of `40`.
  - DeePMD's input documentation says `repinit.nsel`, `repinit.three_body_sel`, and
    `repformer.nsel` can be set as integers or as `auto[:factor]`; it recommends integer
    values below about `200`. Needing `324` neighbors at the 6 A stage is therefore not
    a minor typo. It says this dense/cutoff setup is outside the comfortable range for
    the simple hand-picked DPA-2 selections used here.
- The hand-written DPA-2 config also diverges from the official recommended example.
  DeePMD's `examples/water/dpa2/input_torch_medium.json` includes `use_three_body=true`,
  `three_body_sel=48`, smoother repformer cutoffs (`rcut_smth=3.5` for the 4 A stage),
  `update_style=res_residual`, `update_residual=0.01`, and `gradient_max_norm=5.0`.
  Our config omitted the three-body representation and used a more abrupt smoothing
  setting (`rcut_smth=0.5`) for both stages. That does not prove this is the cause, but
  it means we did not test the official "medium" DPA-2 recipe.
- The DPA-2 force traces did not improve like the `se_e2_a` traces.
  - `variant_train_dpa2_PT`: final force RMSE `3.49`, late-100 median `3.71`.
  - `variant_train_dpa2_PT_big`: final force RMSE `3.49`, late-100 median `3.71`.
  - `variant_train_se_e2_a_PT`: final force RMSE `0.598`, late-100 median `0.352`.
  - `variant_train_se_e2_a_TF`: final force RMSE `0.36`, late-100 median `0.3485`.
- Increasing DPA-2 width changed energy behavior but did not fix the force problem.
  The small and big DPA-2 runs have nearly identical force RMSE values at many logged
  steps, which points to a setup/training issue rather than ordinary capacity scaling.
- The more-neighbor DPA-2 v2 runs were not clean completed models.
  - `variant_train_dpa2_PT_v2` failed near `972800/1000000` steps with a PyTorch
    dataloader/shared-memory failure.
  - `variant_train_dpa2_PT_big_v2` failed near `556100/1000000` steps with the same
    class of failure.
  - `variant_train_dpa2_PT_big_strip_v2` failed near `499900/1000000` steps due to a
    filesystem input/output error while reading `fparam.npy`.
- These runs are not a clean He/MgSiO3 production-quality test.
  The training inputs used `255` `MgSiOH__R2SCAN` systems and did not contain
  `He_MgSiO3` data paths. The neighbor statistics also reported zero type-4 neighbors
  in the `se_e2_a` reference run. This makes the runs useful for setup/runtime behavior,
  but not for judging He interpolation quality.

## Practical Next Test

Run one small, corrected DPA-2 diagnostic before spending more GPU time:

1. Use the actual He/MgSiO3 training split intended for validation, not the MgSiOH
   speed-benchmark collection.
2. Start from the official DeePMD DPA-2 medium example structure, then adapt only the
   data paths/type map and dense-system neighbor settings. In particular, include the
   three-body block and the tuned repformer/update settings unless there is a specific
   reason to deviate.
3. Set DPA-2 neighbor selections using `auto`/`auto:factor` first, or run a startup-only
   neighbor-stat check, so the logs show no `sel is not enough` warnings. If the required
   integer `nsel` is far above DeePMD's recommended `<200` range, treat that as a cost
   and stability warning; consider smaller cutoffs/smoother settings before launching a
   full production run.
4. Keep the baseline fitting net first; do not scale width until the neighbor-selection
   warnings are gone and force RMSE descends normally.
5. Reduce PyTorch data-path fragility for the diagnostic run: use a local copied data
   subset, avoid heavy dataloader/shared-memory pressure, and monitor whether the run
   fails with `Bus error` or filesystem I/O before comparing model quality.
6. Prefer testing both:
   - from-scratch DPA-2 with the corrected official-style config, and
   - fine-tuning from a DPA-2 pretrained model if a compatible type map/checkpoint exists.
7. Judge model quality on held-out energy RMSE/atom for the same `71MgSiO3_5He`
   validation split used for the `se_e2_a` models.

## 2026-05-18 Diagnostic Suite

Submitted four 200k-step MgSiOH diagnostics to isolate the main setup hypotheses before
launching another full 1M-step DPA-2 run:

- `variant_train_dpa2_PT_current_auto_200k`, job `18320018`:
  old hand-written DPA-2 architecture with `auto:1.1` neighbor selections.
- `variant_train_dpa2_PT_doc_medium_auto_currentloss_200k`, job `18320019`:
  official-medium-style DPA-2 with three-body terms and the current energy/force/virial
  loss.
- `variant_train_dpa2_PT_doc_medium_auto_novirial_200k`, job `18320016`:
  official-medium-style DPA-2 with no virial loss.
- `variant_train_dpa2_PT_doc_medium_auto_no3body_200k`, job `18320017`:
  official-medium-style DPA-2 without the three-body block.

All four were pending on `gpuA100x4` with reason `Priority` at the submission check.

## Documentation / Literature Pointers

- DeePMD DPA-2 model docs:
  https://docs.deepmodeling.org/projects/deepmd/en/latest/model/dpa2.html
- DeePMD training-input docs for DPA-2 `repinit.nsel`, `three_body_sel`, and
  `repformer.nsel`:
  https://docs.deepmodeling.org/projects/deepmd/en/master/train/train-input.html
- Official DeePMD DPA-2 medium example:
  https://raw.githubusercontent.com/deepmodeling/deepmd-kit/master/examples/water/dpa2/input_torch_medium.json
- Official DeePMD DPA-2 example README:
  https://raw.githubusercontent.com/deepmodeling/deepmd-kit/master/examples/water/dpa2/README.md
- DPA-2 paper:
  https://www.nature.com/articles/s41524-024-01493-2
- DPA-1 paper, useful background for attention-based Deep Potential pretraining:
  https://arxiv.org/abs/2208.08236
