# Manuscript Source Manifest

This source snapshot was copied from `docs/paper/context/` for JBHI/DBHI submission preparation.

## Files Included

- `main.tex`: main manuscript source.
- `references.bib`: bibliography database.
- `main.bbl`: generated bibliography file, included for source-submission convenience.
- `ieeecolor.cls`: IEEE class/style dependency present in the original manuscript directory.
- `generic.sty`: style dependency present in the original manuscript directory.
- `tables/`: LaTeX table inputs used by `main.tex`.
- `figures/`: figure files referenced by `main.tex`.

## Build Note

From this directory:

```bash
latexmk -pdf main.tex
```

If the journal portal requests source files as a zip, package this directory after removing any local build outputs newly generated during verification.

Verification note: this source snapshot was test-built once with `latexmk -pdf main.tex` on 2026-04-19 and produced an 8-page PDF. Temporary build outputs were removed afterward.
