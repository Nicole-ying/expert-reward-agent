# CREATE - ISCSIC 2026 Session VII submission draft

Target session: **Frontiers of Intelligent Systems: AI-Driven Control, Autonomy, and Emerging Challenges**.

## Files
- `main.tex`: anonymous five-page manuscript source.
- `references.bib`: IEEE-style bibliography database.
- `scripts_generate_figures.py`: regenerates the two vector figures.
- `data/iteration_scores.csv`: scores used to draw the curves.
- `data/seed_summary_frozen.csv`: seed-wise summary used in the manuscript.

## Frozen evidence used
- LunarLander-v3: `runs/env_001/paper_v4`, seeds 0-4.
- BipedalWalker-v3: `runs/env_002/paper_bipedal_main_v1`, seeds 0-4.
- Ablation: `runs/env_001/ablation_unconstrained_v4`, seeds 0-4.
- Repository snapshot audited: `8a6b838c45d0f16dc7fc554197b955a2e13bf274`.

## Excluded claims
The draft deliberately excludes the conflicting 0/5 budget-matched independent-generation claim and excludes Ant-v4 from the abstract and main results. Held-out results are not reported because their raw per-seed artifacts were not verified in the audited result manifest.

## Build
```bash
python scripts_generate_figures.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex8 main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Before CMT upload
1. Download and inspect the official ISCSIC 2026 Full Paper Template ZIP.
2. Confirm CMT topic selection is Session VII, not Session II.
3. Confirm the anonymous PDF remains exactly five pages.
4. Prepare the separate title page required by the conference.
5. Remove repository-identifying metadata from supplementary files during double-blind review.
