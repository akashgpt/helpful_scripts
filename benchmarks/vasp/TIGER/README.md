# VASP benchmark summary for `testing_EXCF`

Analyzed on 2026-04-06.

Source directory: `/scratch/gpfs/BURROWS/akashgpt/qmd_data/testing_EXCF`

## Scope

This report covers the 13 numbered benchmark cases at the top level of `testing_EXCF`:

- `1_PBEsol`
- `1_PBEsol__ALGO_ALL`
- `1_PBEsol__low_NBANDS`
- `1_PBEsol__OG`
- `1_PBEsol__old_config`
- `1_R2SCAN__ALGO_All`
- `1_R2SCAN__ALGO_All_w_ISEARCH`
- `1_R2SCAN__ALGO_F`
- `1_R2SCAN__ALGO_N`
- `1_R2SCAN__ALGO_N__low_NBANDS`
- `1_R2SCAN__ALGO_N__low_NBANDS__old_config`
- `1_R2SCAN_ALGO_VF`
- `1_R2SCAN__PREC`

The top-level `OG/` and `misc/` directories appear to be archival/reference material and were not treated as part of the active sweep.

## High-level findings

- No hard runtime failures were found in the numbered cases.
- All numbered cases except `1_PBEsol__OG` contain `done_RUN_VASP`.
- `1_PBEsol__OG` still looks numerically complete: it has `OSZICAR`, `OUTCAR`, timing information, and final energies, but it lacks the wrapper artifacts (`done_RUN_VASP`, `sub_vasp.sh`) used by the newer runs. I would treat it as a legacy/reference result, not a failed run.
- Multiple `slurm-*.out` files are present in many cases, which indicates reruns or copied job histories, but I did not find obvious scheduler/runtime errors in those logs.
- The `OUTCAR` phrase `kinetic energy error for atom=...` appears in many cases. In this context it is a standard VASP message and should not be interpreted as a benchmark failure.

## Execution environment

From `sub_vasp.sh` in the active cases:

- Scheduler: Slurm
- Allocation: `--account=burrows`
- Resources: 1 node, 112 MPI tasks per node, 1 CPU per task, 1 GB per CPU
- Walltime limit: 5 hours
- Executable path: `/scratch/gpfs/BURROWS/akashgpt/softwares/vasp.6.4.3/bin/vasp_std`
- Modules:
  - `intel-oneapi/2024.2`
  - `intel-mpi/oneapi/2021.13`
  - `intel-mkl/2024.2`
  - `hdf5/oneapi-2024.2/1.14.4`

The newer portable input style uses `NCORE = 8`. The older cluster-tuned inputs instead use `NPAR = 14`, and one PBEsol old-config case also sets `NPACO = 104` and `APACO = 10.4`.

## Setup deltas that were actually tested

### PBEsol family

- `1_PBEsol`: baseline portable setup, `GGA = PS`, `ALGO = VeryFast`, `NBANDS = 1120`
- `1_PBEsol__ALGO_ALL`: same as baseline but `ALGO = All`
- `1_PBEsol__low_NBANDS`: same as baseline but `NBANDS = 560`
- `1_PBEsol__old_config`: low-NBANDS style plus cluster-tuned `NPAR = 14`, `NPACO = 104`, `APACO = 10.4`
- `1_PBEsol__OG`: legacy/reference PBEsol result with `NBANDS = 1120` and no wrapper completion markers

### R2SCAN family

- `1_R2SCAN__ALGO_N`: practical baseline, `METAGGA = R2SCAN`, `LASPH = .TRUE.`, `ALGO = Normal`, `NBANDS = 1120`
- `1_R2SCAN__ALGO_All`: baseline but `ALGO = ALL`
- `1_R2SCAN__ALGO_All_w_ISEARCH`: `ALGO = ALL` plus `ISEARCH = 1`
- `1_R2SCAN__ALGO_F`: baseline but `ALGO = Fast`
- `1_R2SCAN_ALGO_VF`: baseline but `ALGO = VeryFast`
- `1_R2SCAN__ALGO_N__low_NBANDS`: baseline but `NBANDS = 560`
- `1_R2SCAN__ALGO_N__low_NBANDS__old_config`: low-NBANDS style plus cluster-tuned `NPAR = 14`
- `1_R2SCAN__PREC`: `ALGO = ALL` plus `PREC = Accurate`

## Results

### PBEsol cases

| Case                   | Status          |     F (eV) |    E0 (eV) | Last SCF iter | Elapsed (s) | Main interpretation                                     |
| ---------------------- | --------------- | ---------: | ---------: | ------------: | ----------: | ------------------------------------------------------- |
| `1_PBEsol`             | completed       | -861.11413 | -841.69058 |            40 |     854.734 | Portable baseline                                       |
| `1_PBEsol__ALGO_ALL`   | completed       | -861.11413 | -841.69054 |            45 |    1107.244 | Same energy, ~29.5% slower than baseline                |
| `1_PBEsol__low_NBANDS` | completed       | -861.11410 | -841.69077 |            40 |     417.614 | ~51.1% faster than baseline, tiny energy change         |
| `1_PBEsol__OG`         | legacy complete | -861.11410 | -841.69077 |            40 |     343.306 | Looks complete, but lacks wrapper completion markers    |
| `1_PBEsol__old_config` | completed       | -861.11410 | -841.69077 |            40 |     344.758 | ~59.7% faster than baseline with cluster-tuned settings |

Key comparisons against `1_PBEsol`:

- `ALGO = All` did not improve the final energy in a meaningful way and cost about 29.5% more walltime.
- Reducing `NBANDS` from 1120 to 560 cut walltime by about 51.1% while changing `F` by only `+0.000030 eV` and `E0` by `-0.000190 eV`.
- The best walltime came from the old-config/legacy style runs (~344 s), but those rely on cluster-specific parallel settings rather than the more portable `NCORE = 8` style.

### R2SCAN cases

| Case                                       | Status    |     F (eV) |    E0 (eV) | Last SCF iter | Elapsed (s) | Main interpretation                                                |
| ------------------------------------------ | --------- | ---------: | ---------: | ------------: | ----------: | ------------------------------------------------------------------ |
| `1_R2SCAN__ALGO_N`                         | completed | -899.39821 | -881.19800 |            37 |    2499.252 | Best portable 1120-band baseline                                   |
| `1_R2SCAN__ALGO_All`                       | completed | -899.39821 | -881.19805 |            45 |    3669.914 | ~46.8% slower than `ALGO = Normal`                                 |
| `1_R2SCAN__ALGO_All_w_ISEARCH`             | completed | -899.39821 | -881.19805 |            45 |    3627.755 | Slightly faster than `ALGO = ALL`, still much slower than `Normal` |
| `1_R2SCAN__ALGO_F`                         | completed | -899.39821 | -881.19798 |            34 |    3192.729 | ~27.7% slower than `ALGO = Normal`                                 |
| `1_R2SCAN_ALGO_VF`                         | completed | -899.39820 | -881.19800 |            47 |    5003.640 | Much slower; `VeryFast` is a poor choice here                      |
| `1_R2SCAN__ALGO_N__low_NBANDS`             | completed | -899.39820 | -881.19804 |            40 |    1277.602 | ~48.9% faster than `ALGO = Normal`, tiny energy change             |
| `1_R2SCAN__ALGO_N__low_NBANDS__old_config` | completed | -899.39820 | -881.19804 |            40 |     979.174 | ~60.8% faster than `ALGO = Normal` with cluster-tuned settings     |
| `1_R2SCAN__PREC`                           | completed | -899.40018 | -881.19999 |            45 |    7312.245 | Much more expensive; lower energy by about 2 meV                   |

Key comparisons against `1_R2SCAN__ALGO_N`:

- `ALGO = Normal` outperformed the other 1120-band algorithm choices in walltime.
- `ALGO = ALL` and `ALGO = ALL + ISEARCH = 1` converged to essentially the same energy as `ALGO = Normal`, but cost about 45-47% more walltime.
- `ALGO = Fast` was also slower than `Normal`.
- `ALGO = VeryFast` was the worst of the standard algorithm variants here, taking about 100.2% more walltime than `Normal`.
- Reducing `NBANDS` from 1120 to 560 cut walltime by about 48.9% while changing `F` by only `+0.000010 eV` and `E0` by `-0.000040 eV`.
- The old-config low-NBANDS run is the fastest R2SCAN result, but again it depends on cluster-specific parallelization (`NPAR = 14`).
- `PREC = Accurate` lowers the final energy by about `0.00197 eV` relative to `1_R2SCAN__ALGO_N`, but walltime rises by about 192.6%.

## Recommended takeaways

- For portable PBEsol runs on this machine, the strongest cost/performance point in this set is `ALGO = VeryFast` with `NBANDS = 560`.
- For portable R2SCAN runs on this machine, the strongest cost/performance point in this set is `ALGO = Normal` with `NBANDS = 560`.
- If you are willing to use cluster-specific parallel settings, the old-config variants are faster still and preserve essentially the same energies within the resolution shown here.
- Avoid assuming `ALGO = Fast` or `ALGO = VeryFast` will help for R2SCAN. In this sweep they were worse than `ALGO = Normal`.
- Use `PREC = Accurate` only when the extra ~2 meV energy gain is worth a roughly 3x walltime penalty.

## Important files to keep in mind for future benchmarks

For each active case, the most useful files are:

- Inputs: `INCAR`, `POSCAR`, `KPOINTS`, `POTCAR`, `sub_vasp.sh`
- Quick status check: `done_RUN_VASP`, `log.run_sim`, `slurm-*.out`
- Numerical verification: `OSZICAR`, `OUTCAR`
- Post-processing/archive: `vasprun.xml`, `vaspout.h5`, `CONTCAR`, `XDATCAR`, `CHGCAR`, `WAVECAR`

Special notes:

- The legacy `OG/` reference directories do not include the newer wrapper markers in the same way as the active benchmark cases.
- The `misc/1_PBEsol__ALGO_ALL__R2SCAN__ALGO_ALL` directory looks like an extra archived experiment rather than part of the main numbered sweep.

## Files created in this benchmark report directory

- `README.md`: this narrative summary
- `testing_EXCF_summary.tsv`: machine-readable case table
