# Building Knowledge Graphs from Unstructured Text

Challenge: Build a Knowledge Graph for the company **Bayer**, focused on their **Pharmacology business**.

Please refer to the notebook `main.ipynb` for usage and visualizations of our results.

## Installation

```sh
# Create conda environment
conda create -n nlp python=3.7
conda activate nlp

# Install and setup Spacy
conda install -c conda-forge spacy==2.1.6
python -m spacy download en
python -m spacy download en_core_web_lg

# Install neuralcoref (specific version, for spacy compatibility)
conda install cython
curl https://github.com/huggingface/neuralcoref/archive/4.0.0.zip -o neuralcoref-4.0.0.zip -J -L -k
cd neuralcoref
python setup.py build_ext --inplace
python setup.py install

# Install additional packages
pip install wikipedia-api
conda install pandas networkx matplotlib seaborn
conda install pytorch=1.2.0 cudatoolkit=10.0 -c pytorch
pip install transformers

# Install extras
conda install ipywidgets nodejs -c conda-forge
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```
