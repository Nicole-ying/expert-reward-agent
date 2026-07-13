# Paper Analysis Pipeline

This directory contains reproducible result collection and plotting code. Plotting code never hardcodes experiment scores.

## V6 exploratory figures

Run from the repository root:

```powershell
C:\ProgramData\miniconda3\envs\eure\python.exe -m analysis.paper.collect_v6_results
C:\ProgramData\miniconda3\envs\eure\python.exe -m analysis.paper.generate_v6_figures
```

Data products are written to `artifacts/paper/v6_exploratory/`. Figures are written as both PNG and PDF to `figures/paper/v6_exploratory/`.

The v6 dataset is exploratory. Final paper experiments must use a new manifest and a frozen implementation.

## DERES paper figures

Run from the repository root:

```powershell
python -m analysis.paper.generate_deres_core_framework
python -m analysis.paper.generate_deres_paper_figures
```

The script parses current frozen paper logs under `runs/env_001`, `runs/env_002`, and `runs/env_005`, then writes:

- paper-ready PNG/PDF figures to `figures/paper/deres_main/`;
- parsed CSV tables to `figures/paper/deres_main/tables/`;
- a reusable file index to `figures/paper/deres_main/figure_manifest.md`;
- framework reference images copied to `figures/paper/deres_main/framework_reference/`.

The script uses external evaluation scores for all main result figures. Env_005 is treated as a seed_0 case study in the main cross-environment view; all parsed Env_005 records are still preserved in `all_iteration_results.csv` and `method_summary.csv`.
