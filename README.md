# Cropping outperforms dropout as an augmentation strategy for self-supervised training of text embeddings
###### Rita González-Márquez, Philipp Berens & Dmitry Kobak

In this repository you can find the code associated to the paper ["Cropping outperforms dropout as an augmentation strategy for training self-supervised text embeddings"](https://arxiv.org/abs/2508.03453).

## How to use this repository
The notebooks in `scripts/` contain the code to reproduce all the experiments and analyses performed in the paper. The experiments were run in python notebooks that use some helper functions in the `text_embeddings_src/` folder. This was done to make it easier to follow the different steps of the analyses, to be able to visualize intermediate results, and to log hyperparameters used in each experiment directly in the notebook.

The notebook `08-rgm-figures.ipynb` contains the code to generate the final figures included in the paper. All figures generated with the notebooks will be stored in the `figures_path` define at the top (by default `results/figures/updated_dataset/final_figures/last_version`). 

## Installation
This project depends on Python ($\geq$ 3.7). The project script can be installed via `pip install .` in the project root, i.e.:
```
git clone https://github.com/berenslab/text-embed-augm
cd text-embed-augm
pip install -e .
```

## Example of training and evaluations
In notebook `17` you can see an example of how the training and evaluations are done in the notebooks. 
If you are just interested in knowing how specific parts are implemented, you can find:
- **data augmentations** in `data_stuff.py`,
- how the **models are trained** in `train_stuff.py`,
- **evaluations** in `eval_functions.py`,

all of those in the `text_embeddings_src/` folder.

## Equivalence between paper sections and notebooks 

Here is a mapping of the different sections of the paper to the corresponding notebooks where the experiments were run. Numbers correspond to the notebook numbers in the `scripts/` folder.
- **Section 3.2** MTEB performance after SSL fine-tuning
    + **3.2.2** (MTEB eval) : `12`
    + **3.2.3** (per batch) : `17`
- **Section 3.3** Representation of a dataset for analysis and visualization
    + **3.3.2** (kNN accuracy & t-SNE) : `01`, `02`, `03`, `11`, `13`, `14` (ICLR); `15` (Arxiv, Biorxiv, Medrxiv, Reddit, StackExchange).
    + **3.3.3** (sentence and domain adaptation) : `03`, `04`
- **Section 4** SSL without pretraining 
    + `03`, `14`
- **Section 5** Representation quality across layers
    + Fig5a (frozen vs. finetuned) : `05`
    + Fig5b (layer-wise eval kNN): `10`
    + Fig5c (layer-wise eval MTEB) : `18`
    + projection head : `07`
- **Appendix**
    + **A1** Hyperparameter exploration : `06`


## Detailed description of files

Here there is a more detailed description on what you can find in the different notebooks in the `scripts/` folder and the scripts in the `text_embeddings_src/` folder.

#### Scripts in `text-embed-augm/`:
- `data_stuff.py` : data augmentation functions and dataset class.
- `dim_red.py` : dimensionality reduction functions.
- `embeddings.py` : functions to compute embeddings with different models, used in model wrappers.
- `eval_functions.py` : evaluation classes (kNN accuracy, linear classification, MTEB benchmark).
- `metrics.py` : functions to compute different metrics, used in evaluation classes.
- `models.py` : Models and model wrappers for different architectures, and utility functions.
- `plotting.py` : plotting functions.
- `train_stuff.py` : training loop functions.
- `load_mteb_scores_utils.py` : utils to load MTEB results.

This code is the refactored version (current) and the old versions of the source code can be found in the `legacy/` folder. They are kept for reproducibility reasons, since notebooks `01-16` in the `scripts/` folder still use some of those functions. Notebooks `17-18` use the refactored code in the main `text_embeddings_src/` folder.


#### Notebooks in `scripts/`:
- `01-rgm-baseline-embeddings-iclr.ipynb` : obtain high-dimensional representations of the ICLR dataset by the baseline models.
- `02-rgm-baseline-evaluation-iclr.ipynb` : kNN accuracy evaluation of ICLR dataset high-dimensional representations and 2D representations with t-SNE plots.
- `03-rgm-training-models-iclr.ipynb` : fine-tuning models on the ICLR dataset with cropping augmentation, kNN and linear classification accuracy evaluations, also eval after every batch.
- `04-rgm-training-model-other-datasets.ipynb` : sentence vs. domain adaptation experiments on other datasets (Arxiv, Biorxiv and Reddit).
- `05-rgm-exploration-freezing-layers.ipynb` : fine-tuning and evaluation of kNN accuracy when freezing different layers of the model.
- `06-rgm-training-hyperparameters.ipynb` : Hyperparameter exploration for training with different augmentations.
- `07-rgm-training-projection-layer.ipynb` : Fine-tuning models with and without projection layer, evaluation of kNN accuracy.
- `08-rgm-figures.ipynb` : Generate final figures for the paper.
- `09-rgm-whitening.ipynb` : kNN accuracy evaluation after applying centering and whitening to the embeddings.
- `10-rgm-guillotine.ipynb` : kNN accuracy evaluation of every layer for different datasets (ICLR, Reddit, Medrxiv and StackExchange) for both augmentations.
- `11-rgm-simcse-training.ipynb` : fine-tuning models on the ICLR dataset with dropouts augmentation, kNN accuracy evaluation, also eval after every batch.
- `12-rgm-benchmarks.ipynb` : MTEB benchmark evaluation of baseline and fine-tuned models on the ICLR dataset.
- `13-rgm-pretrained-embedding-layer.ipynb` : Training and kNN accuracy evaluation of fine tuning pretrained embedding layer and module with different hyperparameters.
- `14-rgm-random-embedding-layer.ipynb` : Training and kNN accuracy evaluation of fine tuning random initialized embedding layer and module with different hyperparameters. 
- `15-rgm-mteb-datasets-training.ipynb` : Computing high-dimensional representations and kNN accuracy evaluation of other datasets (Arxiv, Biorxiv, Medrxiv, Reddit, StackExchange).
- `16-rgm-mteb-datasets-tsne.ipynb` : Computing 2D representations and kNN accuracy evaluation of other datasets (Arxiv, Biorxiv, Medrxiv, Reddit, StackExchange).
- `17-rgm-mteb-eval-batches.ipynb` : Fine-tuning on the ICLR dataset and evaluation of MTEB benchmark after every batch.
- `18-rgm-mteb-eval-layers.ipynb` : MTEB evaluation of every layer, for both augmentation strategies.
- `print-results.ipynb` : notebook to print the results of the different experiments in a more organized way, and to generate the tables included in the paper.
- `matplotlib_style.txt` : matplotlib RC stylefile for the figures in the paper.









