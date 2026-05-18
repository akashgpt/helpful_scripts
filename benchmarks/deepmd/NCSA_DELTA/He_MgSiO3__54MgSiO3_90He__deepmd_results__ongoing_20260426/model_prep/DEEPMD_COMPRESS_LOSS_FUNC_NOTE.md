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

## Workaround Used

For compression, generate a compression-only copy of the input JSON and remove
the legacy key:

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

The original training input should remain untouched. Use the sanitized
`myinput.compress.json` only for compression.

## Practical Rule

For future ALCHEMY/He_MgSiO3 model-prep scripts:

- Freeze from the trained checkpoint as usual.
- Compress with a strict-parser-compatible training JSON.
- If `dp compress` reports `undefined key loss_func`, remove only
  `loss.loss_func` from a copy of the JSON and retry compression.
- Do not treat this error as a model-quality failure.

