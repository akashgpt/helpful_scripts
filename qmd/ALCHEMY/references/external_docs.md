# External Docs

Use the local `ALCHEMY__dev/README.md` first. Use the links below when you need official software semantics or syntax.

## DeePMD-kit

- Docs landing page: https://docs.deepmodeling.com/projects/deepmd/en/master/index.html
- LAMMPS integration overview: https://docs.deepmodeling.com/projects/deepmd/
- `pair_style deepmd`: https://docs.deepmodeling.com/projects/deepmd/en/v1.3.2/lammps-pair-style-deepmd.html

## LAMMPS

- Manual landing page: https://docs.lammps.org/
- `read_data`: https://docs.lammps.org/read_data.html
- `dump`: https://docs.lammps.org/dump.html
- `units`: https://docs.lammps.org/units.html

## PLUMED

- Manual landing page: https://www.plumed.org/doc.html
- CV documentation: https://www.plumed.org/doc-v2.10/user-doc/html/_colvar.html
- `PRINT`: https://www.plumed.org/doc-v2.10/user-doc/html/_p_r_i_n_t.html

## When These Matter Here

- `collect_plumed_cv_parameters.py` is easier to interpret if you know how PLUMED CVs and `PRINT` behave.
- `refine_deepmd.sh` depends on DeePMD CLI and LAMMPS-DeePMD conventions.
- The ALCHEMY pipeline itself is defined locally, so use the web docs to check syntax, not to replace repo-specific workflow logic.

