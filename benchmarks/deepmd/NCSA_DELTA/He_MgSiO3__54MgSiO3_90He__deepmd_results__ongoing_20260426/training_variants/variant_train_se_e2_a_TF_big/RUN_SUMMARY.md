# `variant_train_se_e2_a_TF_big`

## Curated Result

- Status: complete; finished-training marker was present.
- Backend / descriptor: TensorFlow / `se_e2_a`.
- Target steps: `1000000`.
- Last logged step: `1000000` (`100.0%`).
- Average training time: `0.0805 s/batch`.
- Wall time: `82379.072 s`.
- Parameter count: `23.91 M`.
- Descriptor neurons: `[75,150,300]`.
- Fitting net: `[720,720,720]`.
- Systems: `255`.
- L-curve rows: `10001`.
- Force RMSE signal: final `0.391`, best `0.117`, last-100 median `0.3815`,
  last-100 p10/p90 `0.237` / `0.672`.

## Lesson

This width-only TensorFlow `se_e2_a` case is roughly `9x` larger than base by parameter
count and about `2.1x` slower per batch in this benchmark. Later held-out energy
validation showed it improved over base, but not as much as the balanced width+depth
model.

Raw `log.train`, `lcurve.out`, `slurm-*`, generated `input_v2_compat.json`, and
generated `out.json` were summarized here and removed from the benchmark folder.

