# GraphOBA

Optimal Budget Allocation for Crowdsourcing Labels on Graphs

This repository provides a Python 3.12-compliant, CPU-focused implementation of the **GraphOBA** framework as described in:

> *Optimal Budget Allocation for Crowdsourcing Labels for Graphs* (UAI 2023)

## Features

- Supports experiments on **Cora, Citeseer, Pubmed, WebKB** datasets.
- Loads datasets directly from Hugging Face (`graphbenchmark` repository).
- Binary classification setup with dataset-specific preprocessing.
- Belief Propagation (BP) label propagation engine.
- Implements **GraphOBA-EXP** and **GraphOBA-OPT** policies.
- Train/Test split with 80/20 ratio (random split, reproducible).

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:
- `numpy`
- `scipy`
- `networkx`
- `datasets`
- `scikit-learn`

## Usage

Run experiments via CLI:

```bash
python -m graphoba.main \
    --dataset {cora|citeseer|pubmed|webkb} \
    --policy {exp|opt} \
    --budget <budget_value>
```

Example:

```bash
python -m graphoba.main --dataset cora --policy exp --budget 500
```

## Dataset notes

- **Preprocessing:**
  - `Cora` and `Citeseer`: Class `0` = positive class; all other classes = negative.
  - `Pubmed`: Class `3` = positive class; all other classes = negative.
  - Random split: 80% training nodes / 20% test nodes.

- **Labeling policy:**
  - Only training nodes are labeled.
  - BP propagates labeling information to test set.
  - Final accuracy is reported on test set.

## Dataset License

The license of the Cora dataset is applicable. Refer to [https://relational.fit.cvut.cz/dataset/CORA](https://relational.fit.cvut.cz/dataset/CORA) for license agreement details.

```
@article{mccallum2000automating,
  title={Automating the construction of internet portals with machine learning},
  author={McCallum, Andrew Kachites and Nigam, Kamal and Rennie, Jason and Seymore, Kristie},
  journal={Information Retrieval},
  volume={3},
  pages={127--163},
  year={2000},
  publisher={Springer}
}
```

## Paper Citation

```
@inproceedings{kulkarni2023optimal,
  title={Optimal Budget Allocation for Crowdsourcing Labels for Graphs},
  author={Kulkarni, Adithya and Chakraborty, Mohna and Xie, Sihong and Li, Qi},
  booktitle={Uncertainty in Artificial Intelligence},
  pages={1154--1163},
  year={2023},
  organization={PMLR}
}
```

## Project structure

```
graphoba/
├── __init__.py
├── main.py               # CLI entry point
├── graph.py              # Dataset loading and preprocessing
├── belief_propagation.py # Belief propagation engine
├── policy.py             # GraphOBA-EXP / GraphOBA-OPT policies
├── state.py              # Posterior and marginals state manager
├── worker.py             # Simulated crowd worker responses
├── utils.py              # Utilities (accuracy, seed handling)
├── config.py             # Global config
└── requirements.txt      # Dependencies
```