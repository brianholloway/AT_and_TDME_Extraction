# AT_and_TDME_Extraction

# Supplementary Code and Data for Autler–Townes Analysis

This repository contains the Python analysis scripts and associated NumPy compressed archive (`.npz`) data files used to reproduce selected figures in the accompanying manuscript:

**“Autler--Townes Splitting in Rydberg Atoms: Transition Dipole Matrix Element Extraction and Field Efficiency Analysis.”**

The materials provided here are intended to support transparency and reproducibility of the reported results.

---

## Contents

- `*.py` — Python analysis and plotting scripts  
- `*.npz` — NumPy compressed archive files containing experimentally acquired and processed datasets  
- `figures/` — Directory where regenerated figures are saved

---

## Reproduced Figures

The scripts in this repository reproduce the following figures from the manuscript:

- Figure 6  
- Figures 17–20  

Each script is named according to the figure it generates (e.g. `figure6_analysis.py`).

---

## Usage

All scripts may be executed from the command line using a standard Python environment. For example,

```bash
python -i Fig6_updated.py
