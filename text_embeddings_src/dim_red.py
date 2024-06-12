import numpy as np
from openTSNE import TSNE, affinity

def run_tsne_simple(Z, k=10, rs=42, verbose=False):
    A = affinity.Uniform(
        Z,
        verbose=verbose,
        method="exact",
        random_state=rs,
        k_neighbors=k,
    )

    X = TSNE(
        verbose=verbose, initialization="spectral", random_state=42
    ).fit(affinities=A)

    return X