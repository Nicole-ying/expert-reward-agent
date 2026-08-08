# CREATE ISCSIC 2026 Chapters 1--3 Draft

This package contains an anonymized English LaTeX draft of the title, Introduction, Related Work, and CREATE Method sections, together with the framework figure and verified BibTeX entries.

## Build

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Format note

The ISCSIC 2026 website requires English manuscripts, use of the conference full-paper template, anonymous author information in the main manuscript during review, and at least five pages including figures and references. The official template is distributed as a ZIP archive from the conference website. The project uses the IEEEtran conference class already used by the existing ISCSIC draft and consistent with the conference's IEEE proceedings workflow. Before final submission, compare the compiled manuscript against the downloaded official archive and add the abstract, experiments, conclusion, and required generative-AI disclosure.

## Page-budget note

This package intentionally excludes the abstract, experiments, and conclusion. It is designed to leave approximately two pages of a six-page target for those sections. Page count may change after the remaining sections are inserted.
