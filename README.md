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
conda activate tfglycan-atom
```

## Training and evaluation

```bash
python train.py light --help
python train.py taxonomy --help
python train.py functional --help
python evaluate.py --help
```
