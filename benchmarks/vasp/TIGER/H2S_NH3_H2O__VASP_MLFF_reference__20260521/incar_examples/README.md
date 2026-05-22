# INCAR Examples

These files are small, source-copied examples from the H2S/NH3/H2O VASP MLFF campaign. They are included because the `INCAR` settings are the key reusable part of this benchmark reference.

## Files

| File | Source system | Source path | Purpose |
| --- | --- | --- | --- |
| `INCAR__pure_H2O_250_molecules_b__NPT_train_R2SCAN_MLFF` | Pure H2O, 250 molecules, later `_b` training box | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data/pure_H2O/250_molecules_b/INCAR` | Pure-fluid NPT MLFF training example at `PSTRESS = 200` kbar with `ISIF = 3`, `MDALGO = 3`, `ML_MB = 4000`, and `ML_MCONF = 600`. |
| `INCAR__H2O_NH3_250_H2O_256_NH3_b__two_phase_NVT_train_R2SCAN_MLFF` | Two-phase H2O/NH3, 250 H2O plus 256 NH3, `_b` run | `/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2S_NH3_H2O/sim_data/H2O_NH3/250_H2O__256_NH3__b/INCAR` | Mixed fixed-cell MLFF training example with `ISIF = 2`, `POTIM = 0.25`, `ML_MB = 16000`, and `ML_MCONF = 2000`. |

## How To Read These Examples

Both examples share the same core electronic setup: `METAGGA = R2SCAN`, `ENCUT = 500.00 EV`, `ALGO = N`, `LREAL = A`, `LASPH = .TRUE.`, `NPAR = 14`, and MLFF training mode with `ML_LMLFF = .TRUE.` and `ML_MODE = train`.

The pure-fluid file is the template to inspect when preparing pressure-equilibrated single-component liquids. The two-phase file is the template to inspect when running the stitched mixed liquid cell after the pure components have supplied reasonable starting structures and MLFF state.

