# Knowledge Graphs

## Installation

```sh
conda create -n nlp python=3.7
conda activate nlp

conda install -c conda-forge spacy==2.1.6
python -m spacy download en
python -m spacy download en_core_web_lg

conda install cython
curl https://github.com/huggingface/neuralcoref/archive/4.0.0.zip -o neuralcoref-4.0.0.zip -J -L -k
cd neuralcoref
python setup.py build_ext --inplace
python setup.py install

pip install wikipedia-api

conda install pandas networkx matplotlib seaborn

conda install pytorch=1.2.0 cudatoolkit=10.0 -c pytorch

pip install transformers
conda install ipywidgets nodejs -c conda-forge
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```
