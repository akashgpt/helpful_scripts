# se_e2_a TF Depth-Scaling Variants

These variants are TF-only and start from `shared/train_se_e2_a.json`, not from the wider `big` config.
The purpose is to separate depth scaling from the previous width scaling benchmark.

Baseline trainable network-parameter accounting:

- descriptor `[25, 50, 100]`: `161,250` params
- fitting `[240, 240, 240]`: `2,504,415` params
- total: `2,665,665` params

Each extra descriptor layer appends one `100`-wide layer and adds `252,500` params.
Each extra fitting layer appends one `240`-wide layer and adds `290,400` params.

| target | variant | descriptor | fitting | descriptor params | fitting params | total params | ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| ~2x | `variant_train_se_e2_a_TF_desc_deep2x` | `[25, 50, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]` | `[240, 240, 240]` | 2,938,750 | 2,504,415 | 5,443,165 | 2.042x |
| ~2x | `variant_train_se_e2_a_TF_fit_deep2x` | `[25, 50, 100]` | `[240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240]` | 161,250 | 5,118,015 | 5,279,265 | 1.980x |
| ~2x | `variant_train_se_e2_a_TF_both_deep2x` | `[25, 50, 100, 100, 100, 100, 100, 100]` | `[240, 240, 240, 240, 240, 240, 240, 240]` | 1,423,750 | 3,956,415 | 5,380,165 | 2.018x |
| ~10x | `variant_train_se_e2_a_TF_desc_deep10x` | `[25, 50, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]` | `[240, 240, 240]` | 24,148,750 | 2,504,415 | 26,653,165 | 9.999x |
| ~10x | `variant_train_se_e2_a_TF_fit_deep10x` | `[25, 50, 100]` | `[240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240]` | 161,250 | 26,607,615 | 26,768,865 | 10.042x |
| ~10x balanced | `variant_train_se_e2_a_TF_balanced_10x` | `[60, 120, 240, 240, 240, 240]` | `[620, 620, 620, 620, 620, 620]` | 5,250,000 | 21,554,305 | 26,804,305 | 10.055x |
| ~10x | `variant_train_se_e2_a_TF_both_deep10x` | `[25, 50, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]` | `[240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240]` | 11,271,250 | 15,282,015 | 26,553,265 | 9.961x |

Notes:

- `*_2x` cases roughly double total trainable network params using depth only.
- `*_10x` cases target roughly 10x total trainable network params using depth only.
- The existing `variant_train_se_e2_a_TF_big` is the width-scaling comparison point; its trainable network count was about `23.91 M`, or about `8.97x` the original small model.
- None of these jobs were submitted when this note was written.

Follow-up TODO:

- Create analogous DPA2 depth-scaling cases later, after the TF `se_e2_a` depth-scaling jobs finish or at least establish useful runtime/memory behavior.

Balanced 10x case:

- `variant_train_se_e2_a_TF_balanced_10x` scales both depth and width from the original small model.
- It is intended as the direct comparison against width-only `variant_train_se_e2_a_TF_big` and depth-only 10x cases.
