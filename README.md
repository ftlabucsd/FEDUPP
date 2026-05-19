# FEDUPP — FED3 Users Processing Package

Official implementation for the paper *"The development of FEDUPP: Feeding Experimentation Device Users Processing Package to Assess Learning and Cognitive Flexibility"* from Nature Translational Psychiatry.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/) &nbsp; [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) &nbsp; [![Paper](https://img.shields.io/badge/Nature-paper-00A499.svg)](https://www.nature.com/articles/s41398-026-04091-6) &nbsp; [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20277476-blue.svg)](https://doi.org/10.5281/zenodo.20277476)

> A reusable analysis pipeline for FED3 behavioral data — learning acquisition, cognitive flexibility, and feeding patterns in mice.

---

## Table of Contents

- [Quick Start](#-quick-start)
- [Pipeline Overview](#-pipeline-overview)
- [Detailed Documentation](Overview.md)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [Contact & Support](#-contact--support)
- [License](#-license)
- [Changelog](#-changelog)

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/ftlabucsd/FEDUPP
cd FEDUPP

# Recommended: create an isolated environment
conda create -n fedupp python=3.10
conda activate fedupp
pip install -r requirements.txt
```

If you prefer to use an existing environment, open `pipeline.ipynb` — the first cell checks for missing or mismatched packages and offers a one-line fix.

To launch Jupyter:

```bash
pip install jupyterlab ipykernel
python -m ipykernel install --user --name fedupp --display-name "Python (fedupp)"
jupyter lab
```

> **Replication only?** Stop here and run all cells in `pipeline.ipynb`.
> Continue below only if you want to analyze **your own data**.

### 2. Prepare Your Data

**A. Organize FED3 CSV files** — one subfolder per mouse inside `sample_data/`:

```
sample_data/
├── M1/
│   ├── fr1.csv
│   └── reversal.csv
├── M2/
│   └── fr1.csv
└── ...
```

> Filenames do not need to encode the session type; the pipeline auto-detects FR1 vs. Reversal.

**B. Define group membership** in `group_map.json` (subfolder names must match):

```json
{
  "control": ["M1", "M2", "M3"],
  "experimental": ["M10", "M11", "M12"]
}
```

### 3. Run the Pipeline

Open `pipeline.ipynb` and run cells sequentially. In **Step 2**, set `MEAL_METHOD`:

| Value | Description |
|-------|-------------|
| `'paper'` (default) | Method described in our paper |
| `'ipi'` | Original FED3 paper method |

---

## 📋 Pipeline Overview

The notebook is organized into **24 steps** across four parts:

| Part | Steps | Scope |
|------|-------|-------|
| **Setup** | 1–3 | Import libraries, load data, quality-control dispenser hardware |
| **A — FR1** | 4–8 | Learning curves, meal patterns, ML meal classification, accuracy distributions |
| **B — Reversal** | 9–21 | Block transitions, WSLS strategies, learning scores, retrieval times, cross-feature correlations, combined meal summaries |
| **C — IPI** | 22–23 | Inter-pellet interval analysis by pellet position |
| **D — Export** | 24 | All metrics → multi-sheet Excel file |

For a detailed walkthrough of every step and script module, see **[Overview.md](Overview.md)**.

---

## 📖 Citation

If you use FEDUPP in your research, please cite:

Yao, M., Libster, A.M., Desfor, S. et al. The development of FEDUPP: feeding experimentation device users processing package to assess learning and cognitive flexibility. Transl Psychiatry (2026). https://doi.org/10.1038/s41398-026-04091-6

```bibtex
@article{Yao2026FEDUPP,
  author  = {Yao, Mingyang and Libster, Avraham M. and Desfor, Shane and Malhotra, Freiya and Castorena, Nathalia and Montilla-Perez, Patricia and Telese, Francesca},
  title   = {The development of FEDUPP: feeding experimentation device users processing package to assess learning and cognitive flexibility},
  journal = {Translational Psychiatry},
  year    = {2026},
  doi     = {10.1038/s41398-026-04091-6},
  url     = {https://www.nature.com/articles/s41398-026-04091-6},
  note    = {Mingyang Yao and Avraham M. Libster contributed equally. Published 16 May 2026.}
}
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Follow [PEP 8](https://peps.python.org/pep-0008/), add type hints and docstrings
4. Test on Python 3.10+ with sample data
5. Submit a pull request with a clear description

---

## 📧 Contact & Support

**Maintainer**: [FT Lab](https://www.teleselab.com/)
**Issues**: [GitHub Issues](https://github.com/ftlabucsd/FEDUPP/issues)

For bug reports, include the error traceback and (if possible) the data file that triggers it.

---

## 📜 License

© 2025 FT Lab — [MIT License](LICENSE)

---

## 🔄 Changelog

### v3.0 (November 2025)
- Inter-Pellet Interval (IPI) analysis for feeding rhythm
- Full data export to Excel
- Meal accuracy vs. dispense time correlation

### v2.0 (August 2025)
- Complete pipeline reorganization with modular scripts
- ML-based meal quality classification (LSTM/CNN)
- Enhanced reversal learning analysis (block transitions, retrieval times)
- Automated quality-control checks and built-in statistical testing

### v1.0 (2024)
- Initial release with CASK experiment analysis

---

## 🙏 Acknowledgments

- **FED3 Device**: [Kravitz Lab Open Source Hardware](https://github.com/KravitzLabDevices/FED3)
- **Community Contributors**: Thanks to all researchers who provided feedback

---

**⭐ If this project helps your research, please star the repository!**
