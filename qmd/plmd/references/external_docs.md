# External Docs

## PLUMED

- Manual landing page: https://www.plumed.org/doc.html
- CV documentation: https://www.plumed.org/doc-v2.10/user-doc/html/_colvar.html
- Collective variable overview: https://www.plumed.org/doc-v2.9/user-doc/html/colvarintro.html
- `PRINT`: https://www.plumed.org/doc-v2.10/user-doc/html/_p_r_i_n_t.html
- `DISTANCE`: https://www.plumed.org/doc-v2.10/user-doc/html/_d_i_s_t_a_n_c_e.html

## Why These Matter Here

- `plot_plmd_COLVAR.py` assumes the standard `COLVAR` header and field layout written by PLUMED `PRINT`.
- `plumed.info` summaries are often used to choose or validate CV ranges, so the CV docs help interpret those ranges.
- For repo-specific CV-range harvesting in ALCHEMY, also read `qmd/ALCHEMY/collect_plumed_cv_parameters.py`.

