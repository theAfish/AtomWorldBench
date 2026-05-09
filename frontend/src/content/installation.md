# Installation

## Lightweight install

Just the evaluator — minimal dependencies:

```bash
pip install git+https://github.com/theAfish/AtomWorldBench.git
```

## Full toolkit

CLI, inference, data generation, plotting, etc.:

```bash
pip install "atomworld[all] @ git+https://github.com/theAfish/AtomWorldBench.git"
```

## Development install

```bash
git clone https://github.com/theAfish/AtomWorldBench.git
cd AtomWorldBench
pip install -e ".[dev]"
```

## Optional dependency groups

| Extra          | What it adds                                |
|----------------|---------------------------------------------|
| `[benchmark]`  | openai, pandas, h5py, tqdm, pyyaml          |
| `[datagen]`    | ase, mp-api, scipy, pandas                   |
| `[models]`     | transformers, sentencepiece, torch            |
| `[all]`        | All of the above + ray                        |
| `[dev]`        | `[all]` + pytest                              |