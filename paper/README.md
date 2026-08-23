# Paper

Release files:

- `main.tex` — paper source;
- `references.bib` — verified bibliography;
- `paper.pdf` — compiled eight-page release PDF;
- `figures/` — publication figures;
- `make_publication_figures.py` — deterministic figure regeneration from frozen
  CSV results.

Compile with Tectonic:

```bash
cd paper
tectonic main.tex
```

To regenerate figures first, run the following from the repository root after a
completed experiment has produced `outputs/final_v3/`:

```bash
python paper/make_publication_figures.py
```

The figure script does not run models or alter experimental outputs.
