# TFGlycan

Task-family specialization for glycan prediction.

TFGlycan uses specialized branches for glycan taxonomy, glycan function and protein-glycan interaction.

| Branch | Tasks |
| --- | --- |
| Taxonomy | Domain, kingdom, phylum, class, order, family, genus, species |
| Glycan function | Immunogenicity and glycosylation type |
| Interaction | Protein-glycan binding affinity |

## Installation

Create the environment on Linux:

```bash
conda env create -f env/environment.yml
conda activate tfglycan
```

The all-atom branch uses the [GlycanAA](https://github.com/kasawa1234/GlycanAA) implementation as an external dependency:

```bash
git clone https://github.com/kasawa1234/GlycanAA.git dependencies/GlycanAA
```

Set `GLYCANAA_ROOT` to use an existing installation. Graph extensions require a C++ compiler and a compatible CUDA Toolkit.

## Training and evaluation

```bash
python train.py light --help
python train.py taxonomy --help
python train.py functional --help
python evaluate.py --help
```

Training and model parameters are supplied through the command line. Taxonomy training starts from a lightweight checkpoint. The functional branch supports initialization from an existing checkpoint and PCGrad optimization.

The CSV files are in `data/glycanml/`. Training and evaluation share a data loader that reads a local protein feature file through `--protein-features`: a PyTorch dictionary with `sequences` and `features` entries. Paths are relative to the repository. Set the result directory with `--output-dir`.

## Code

- `src/tfglycan/light.py`: data loading, lightweight encoder and joint training.
- `src/tfglycan/taxonomy.py`: taxonomy specialization.
- `src/tfglycan/atom.py`: all-atom branch and PCGrad training.
- `src/tfglycan/prediction.py`: branch predictions and feature caching.
- `src/tfglycan/evaluate.py`: task-family evaluation and score fusion.
- `src/tfglycan/backend.py`: external encoder interface.
