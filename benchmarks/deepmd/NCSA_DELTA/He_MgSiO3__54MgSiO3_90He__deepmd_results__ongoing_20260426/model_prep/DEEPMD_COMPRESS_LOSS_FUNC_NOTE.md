# DeePMD `dp compress` and Legacy `loss_func`

## Context

During the 2026-05-18 compressed validation pass for the
`71MgSiO3_5He` held-out systems, the first submitted array failed during
`dp compress`.

The workflow was:

```bash
dp freeze -o pv.pb
dp compress -i pv.pb -o pv_comp.pb -t myinput.json
dp test -m pv_comp.pb -s <system> -n 0 -d dp_test
```

Compression failed before any useful tabulation or validation because the
container-side DeePMD parser rejected a legacy key in the training JSON:

```text
dargs.dargs.ArgumentKeyError:
[at location `loss`] undefined key `loss_func` is not allowed in strict mode.
```

## What Happened

The authored He/MgSiO3 training inputs checked in this benchmark do not appear
to intentionally use `loss.loss_func`. They use the normal DeePMD loss block:

```json
"loss": {
  "start_pref_e": 0.04,
  "limit_pref_e": 2,
  "start_pref_f": 1000,
  "limit_pref_f": 1.5,
  "start_pref_v": 0.04,
  "limit_pref_v": 2,
  "type": "ener"
}
```

However, DeePMD-generated normalized outputs such as `out.json` can include:

```json
"loss_func": "mse"
```

If such a generated or normalized JSON is reused as the `-t/--training-script`
input to `dp compress`, newer DeePMD versions may reject it in strict parser
mode.

This is a metadata/parser compatibility issue, not evidence that the checkpoint
or frozen model is bad.

## Important Update

The first attempted workaround was to generate a compression-only copy of the
input JSON and remove the legacy key:

```bash
python -c 'import json, sys
source_path, output_path = sys.argv[1:3]
with open(source_path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
data.get("loss", {}).pop("loss_func", None)
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")' myinput.json myinput.compress.json

dp compress -i pv.pb -o pv_comp.pb -t myinput.compress.json
```

That was not sufficient for the validation-array workflow submitted on
2026-05-18. Even when `myinput.compress.json` contained no `loss_func`,
`dp compress` still failed with the same strict-parser error. This indicates
that, in this DeePMD/container path, the parser may be reading normalized
training metadata from the frozen graph or generated checkpoint metadata rather
than only the JSON passed with `-t`.

By contrast, the working ALCHEMY/He_MgSiO3 training jobs use this simpler
post-training flow inside the real training `model-compression` directory:

```bash
cd model-compression
dp freeze -o pv.pb
dp compress -i pv.pb -o pv_comp.pb
```

They do not pass `-t`. DeePMD then generates/uses its own
`compress.json` and `input_v2_compat.json` in the compression directory. The
checked successful He_MgSiO3 examples under
`/work/nvme/bguf/akashgpt/qmd_data/He_MgSiO3/sim_data_ML/v1_i1/train` and
`/work/nvme/bguf/akashgpt/qmd_data/He_MgSiO3/sim_data_ML/v1_i2/train` have no
`loss_func` in those generated compression inputs and produced `pv_comp.pb`.

The second difference is the DeePMD executable. The successful qmd_data jobs
trained and compressed with the Apptainer image reporting DeePMD `v3.1.3`
commit `b2c8511e`. The benchmark variants in
`testing__LAMMPS__kokkos_bench/.../training_bench/variant_train_se_e2_a_TF*`
were trained in `ALCHEMY_env`, which reported DeePMD `v3.1.3-29-gefc27cf7`.
Those newer generated `out.json` files include `loss.loss_func`, and compression
with the older Apptainer image rejects that metadata. Therefore the practical
compatibility rule is: freeze, compress, and test these benchmark TF models with
the same DeePMD environment used for training, or retrain/regenerate metadata
with the target compression environment.

## Practical Rule

For future ALCHEMY/He_MgSiO3 model-prep scripts:

- Freeze from the trained checkpoint as usual.
- Prefer the ALCHEMY-style native compression path: run `dp freeze` and
  `dp compress -i pv.pb -o pv_comp.pb` inside the checkpoint-containing
  `model-compression` directory, without `-t`.
- Keep DeePMD versions consistent across train/freeze/compress/test. For these
  benchmark TF variants, use `ALCHEMY_env`, because that is the environment that
  created the checkpoint metadata.
- Avoid reusing old frozen `pv.pb` files after a failed compression attempt;
  regenerate the frozen model in a clean compression directory.
- Treat a sanitized `-t` JSON as an experimental fallback only, not the primary
  workflow.
- Do not treat this error as a model-quality failure.

## Separate `big` GraphDef Failure

The original `big` TensorFlow `se_e2_a` variant later failed for a different
reason after the environment mismatch was fixed. With `ALCHEMY_env`, compression
passed the `loss_func` parser stage but crashed while TensorFlow exported the
compressed checkpoint meta graph:

```text
google.protobuf.message.DecodeError: Error parsing message with type 'tensorflow.GraphDef'
```

This happened after compressed checkpoint data and index files were written, but
before `model.ckpt.meta` was emitted. The likely cause is the compressed graph
approaching or crossing a TensorFlow/Protobuf GraphDef serialization-size limit.
The successful `big5x` intermediate already emits a compressed meta graph of
about `1.8G`; the original `big` architecture is wider (`descrpt` neurons
`[75, 150, 300]`, fitting net `[720, 720, 720]`) and plausibly exceeds the
practical meta-graph limit with the default compression step.

Practical remedies to test:

- Retry original `big` with a coarser compression table, for example
  `dp --tf compress -i pv.pb -o pv_comp.pb -s 0.02`, which should reduce table
  and meta-graph size at the cost of a small compression-accuracy check.
- Prefer the successfully compressed intermediate width variants (`big2x`,
  `big5x`) unless original `big` is specifically needed.
- The prior non-compressed `big` `dp_test` numbers may be listed as a reference
  comparison, but mark them explicitly as `noncompressed_reference`; the
  compressed `big` run never reached `dp test`.
