# TFGlycan

Task-family specialization for glycan prediction.

TFGlycan uses specialized branches for glycan taxonomy, glycan function and protein-glycan interaction.

This development version uses an explicit-hydrogen molecular graph, bond-gated message passing, recurrent atom updates and attention pooling. Its atom encoder is implemented in this repository using PyTorch. GlyLES and RDKit provide chemical conversion and atom features.

The atom architecture requires its own training runs and checkpoints. No benchmark results are reported for this version.

| Branch | Tasks |
| --- | --- |
| Taxonomy | Domain, kingdom, phylum, class, order, family, genus, species |
| Glycan function | Immunogenicity and glycosylation type |
| Interaction | Protein-glycan binding affinity |

## Installation

Create the environment on Linux:

```bash
conda env create -f env/environment.yml
conda activate tfglycan-atom
```

## Training and evaluation

```bash
python train.py light --help
python train.py taxonomy --help
python train.py functional --help
python evaluate.py --help
```
