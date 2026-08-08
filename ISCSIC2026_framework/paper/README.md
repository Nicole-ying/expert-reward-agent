# CREATE — ISCSIC 2026 Paper

This directory contains the complete LaTeX source for the five-page ISCSIC
2026 submission draft:

> **CREATE: A Closed-Loop Reward Editing Agent with Training Evidence for
> Reinforcement Learning**

## Contents

- `main.tex` — complete IEEE conference manuscript
- `main.pdf` — compiled five-page review copy
- `figures/` — publication figures required by `main.tex`
- `figures/framework_editable.svg` — editable source for the main framework
- `submission_materials/title_page_template.docx` — conference title-page template

## Build

Compile twice so that references and cross-references are resolved:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The manuscript uses `IEEEtran` and standard TeX packages only. Bibliographic
entries are embedded in `main.tex`, so no separate BibTeX step is required.

## Notes

- The generated reward is used for PPO training.
- The unchanged environment reward is used for native evaluation.
- The main framework is supplied as both a publication-ready vector PDF and an
  editable SVG.
- Experimental plots are stored as vector PDFs to preserve print quality.
