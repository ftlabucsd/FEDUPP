# FEDUPP - FED3 Users Processing Package

This is the official implementation of the paper "The development of FEDUPP: Feeding Experimentation Device Users Processing Package to Assess Learning and Cognitive Flexibility".

 [![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/) &nbsp; [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) &nbsp; [![Paper Link](https://img.shields.io/badge/bioarxiv-paper-red.svg)](https://www.biorxiv.org/content/10.1101/2025.08.14.670424v1)

> **A comprehensive, reusable analysis pipeline for FED3 behavioral data to assess learning acquisition, cognitive flexibility, and feeding patterns in mice.**

---

## 🚀 Quick Start

### 1. Installation

Please clone the repository and goes to the project directory first:

```bash
# Clone the repository
git clone https://github.com/ftlabucsd/FEDUPP
cd FEDUPP
```

If you know how to deal with a conda environment, please follow the typical conda environment installation commands in the terminal below:

```bash
# Create conda environment and install dependencies (requires Python ≥3.10)
conda create -n fedupp python=3.10
conda activate fedupp
pip install -r requirements.txt
```

If you are not so familiar with conda environement, but have one, please directly use this environment and then move to `pipeline.ipynb` and the first cell there will check if your environment works for our project and we provide the package installation commands in the notebook.

You can use the following commands in the terminal to install, run the Jupyter environment, and visit [http://localhost:8888/lab](http://localhost:8888/lab) (if you do not use other IDEs):

```bash
pip install jupyterlab ipykernel
python -m ipykernel install --user --name fedupp --display-name "Python (fedupp)"
jupyter lab
```

**Notice:** If you only want to replicate our analysis and results, you can stop here and directly go to `pipeline.ipynb` and run all cells after you create the environment. **Go to the steps below only if you want to run your own data. **


### 2. Data Preparation

**A. Organize Your Data**

Delete old files/sub-folders inside `sample_data/` and place your FED3 CSV files in it with this structure:

```
sample_data/
├── M1/
│   ├── fr1.csv
│   └── reversal.csv
├── M2/
│   └── fr1.csv
├── M3/
│   ├── fr1.csv
│   └── reversal.csv
└── ...
```
> Note: you do not have to specify the behavioral session type (e.g. FR1 or reversal) in csv filenames. Our algorithm will automatically determine its session. The names is for illustration only, but you *DO* have to put csv files for one mouse in one folder.

**B. Define Group Membership**

Create or modify `group_map.json` to assign mice id to experimental groups, for example like below (the ID you enter here must match the subfolder name, like "M1", "M2" above, but the group name can be customized as desired):

```json
{
  "control": ["M1", "M2", "M3"],
  "experimental": ["M10", "M11", "M12"],
  "validation": ["M20", "M21", "M22"]
}
```

### 3. Run the Analysis Pipeline

Inside this project, open `pipeline.ipynb` in Jupyter Lab or VS Code or other IDEs and run cells sequentially:

The notebook will **automatically**:
1. Load and validate your data
2. Perform quality control checks
3. Generate FR1 and Reversal Learning analyses
4. Save figures to `figures/FR1/` and `figures/REV/`
5. Print statistical test results

---

## 🧠 Detailed Introduction and Documentation

For detailed workflows of the notebook and descriptions of functions FEDUPP includes, please [Read more details here](Overview.md)

---

## 📖 Citation

If you use FEDUPP in your research, please cite:

```bibtex
@article{FEDUPP,
  author  = {Mingyang Yao and Avraham M. Libster and Shane Desfor and Freiya Malhotra and Nathalia Castorena and Patricia Montilla-Perez and Francesca Telese},
  title   = {The development of FEDUPP: Feeding Experimentation Device Users Processing Package to Assess Learning and Cognitive Flexibility},
  year    = {2025},
  journal = {bioRxiv},
  url     = {https://www.biorxiv.org/content/early/2025/08/20/2025.08.14.670424},
  note    = {Mingyang Yao and Avraham M. Libster contributed equally.}
}
```

---

## 🤝 Contributing

We welcome contributions! To contribute:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**:
   - Add new metrics to appropriate script module
   - Update notebook with usage examples
   - Add docstrings to let users quickly know the input and expected output
4. **Test thoroughly** with sample data
5. **Submit a pull request** with clear description

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints to function signatures
- Document all public functions with docstrings
- Test on Python 3.10+ environments
- Keep notebooks cell-by-cell executable

---

## 📧 Contact & Support

**Maintainer**: [FT Lab](https://www.teleselab.com/) <br>
**Report Issues**: [GitHub Issues](https://github.com/your-username/FED3-data/issues)  

For bug reports, include:
- Error message / traceback
- Sample data or file that triggers the error (if possible)

---

## 📜 License

© 2025 FT Lab

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔄 Changelog

### v2.0 (January 2025)
- ✨ Complete pipeline reorganization with modular scripts
- 📖 Comprehensive inline documentation in notebooks
- 🔧 Moved all functions to `scripts/` for reusability
- 🎨 Improved visualization consistency (SVG outputs)
- 🧠 Added ML-based meal quality classification
- 📊 Enhanced reversal learning analysis (block transitions, retrieval times)
- 🧪 Automated quality control checks
- 📈 Statistical testing built into workflow

### v1.0 (2024)
- Initial release with CASK experiment analysis
- Basic FR1 and reversal learning support

---

## 🙏 Acknowledgments

- **FED3 Device**: [Kravitz Lab Open Source Hardware](https://github.com/KravitzLabDevices/FED3)
- **Community Contributors**: Thanks to all researchers who provided feedback and suggestions

---

## 📚 Additional Resources

- [FED3 Hardware Documentation](https://github.com/KravitzLabDevices/FED3)
- [FED3 User Guide](https://github.com/KravitzLabDevices/FED3/wiki)

---

**⭐ If this project helps your research, please give it a star on GitHub!**

