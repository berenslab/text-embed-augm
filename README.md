# Cropping outperforms dropout as an augmentation strategy for training self-supervised text embeddings
###### Rita González-Márquez, Philipp Berens & Dmitry Kobak

In this repository you can find the code associated to the paper ["Cropping outperforms dropout as an augmentation strategy for training self-supervised text embeddings"](https://arxiv.org/abs/2508.03453).

## How to use this repository
The notebooks in `scripts/` contain the code to reproduce all the experiments and analyses performed in the paper.
The notebook `08-rgm-figures.ipynb` contains the code to generate the final figures included in the paper. All figures generated with the notebooks will be stored in the `results/figures/updated_dataset/final_figures/last_version` folder. 

## Installation
This project depends on Python ($\geq$ 3.7). The project script can be installed via `pip install .` in the project root, i.e.:
```
git clone https://github.com/berenslab/text-embed-augm
cd text-embed-augm
pip install -e .
```


