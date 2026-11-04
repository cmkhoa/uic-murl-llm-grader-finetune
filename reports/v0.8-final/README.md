# v0.8 final reproducibility artifacts

Run from the repository root:

```bash
python3 scripts/v08_reproducibility.py
```

This offline command reads archived validation and training bundles only. It never opens a held-out test split or test-results archive. It regenerates the archive manifest, independent metrics, selection decision, slice/confusion tables, and all PNG/SVG figures in this directory. Python 3.10+, NumPy, and Matplotlib are required.

The narrative report is [`../v0.8-final-cross-family-report.md`](../v0.8-final-cross-family-report.md), the notebook interface is [`../../notebooks/v0.8-final-reproducibility.ipynb`](../../notebooks/v0.8-final-reproducibility.ipynb), and its dependency-free static inspection render is [`notebook-render.html`](notebook-render.html).
