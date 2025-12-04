import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import seaborn as sns
#from scvi.external import MRVI
from mrvi_torch import TorchMRVI as MRVI

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("px_hidden", type=int, default=32)
parser.add_argument("px_layers", type=int, default=1)
parser.add_argument("qz_hidden", type=int, default=32)
parser.add_argument("qz_layers", type=int, default=1)

args = parser.parse_args()


scvi.settings.seed = 0  # optional: ensures reproducibility
#print("Last run with scvi-tools version:", scvi.__version__)
#save_dir = tempfile.TemporaryDirectory()
save_dir = '~/mrvi-reproducibility/tmp'

adata_path = os.path.join(save_dir, "haniffa_tutorial_subset.h5ad")

adata = sc.read(adata_path, backup_url="https://figshare.com/ndownloader/files/46017615")
sc.pp.highly_variable_genes(
    adata, n_top_genes=10000, inplace=True, subset=True, flavor="seurat_v3"
)

sample_key = "patient_id"  # target covariate
# batch_key="Site"  # nuisance variable identifier
#MRVI.setup_anndata(adata, sample_key=sample_key, backend="torch")
MRVI.setup_anndata(adata, sample_key=sample_key)

px_kwargs={
    "n_hidden": args.px_hidden,
    "n_layers": args.px_layers
}
qz_kwargs={
    "n_hidden": args.qz_hidden,
    "n_layers": args.qz_layers
}

model = MRVI(adata, px_kwargs=px_kwargs, qz_kwargs=qz_kwargs)

model.train(max_epochs=100)
model.save(f"models/mrvi_model-{args.px_hidden}_{args.px_layers}_{args.qz_hidden}_{args.qz_layers}", overwrite=True, save_anndata=True)