# Cropping outperforms dropout as an augmentation strategy for training self-supervised text embeddings
In this repository you can find the code associated to the paper "Cropping outperforms dropout as an augmentation strategy for training self-supervised text embeddings"

## How to use this repository
The notebooks in `scripts/` contain the code to reproduce all the experiments and analyses performed in the paper. In particular, the notebooks `02-03` contain all the steps for generating the 2D embedding from raw text. The notebook `21-rgm-figures.ipynb` contains the code to generate the final figures included in the paper. All figures generated with the notebooks will be stored in the `results/figures/updated_dataset/final_figures/last_version` folder. 

## Installation
This project depends on Python ($\geq$ 3.7). The project script can be installed via `pip install .` in the project root, i.e.:
```
git clone https://github.com/ritagonmar/text-embeddings
cd text-embeddings
pip install -e .
```


