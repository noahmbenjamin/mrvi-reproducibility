Conda setup
```
conda create --name myenv python=3.11
conda activate myenv

pip install scvi-tools
pip install ipykernel
python -m ipykernel install --user --name myenv --display-name "Kernel Name"

Activate your kernel in the jupyter notebook and run it
```