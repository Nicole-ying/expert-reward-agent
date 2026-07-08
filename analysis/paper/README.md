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
