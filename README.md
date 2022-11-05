# CS5242-ABSC
[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.9-blue)](https://pypi.org/project/autogluon/)

In this project, we look into the market for condominiums in Singapore. We aim to predict the sale prices through data mining techniques, different machine learning models and AutoML frameworks. [Github Link](https://github.com/YanZehong/CS5242-ABSC)

## Table of Codes
- Data Collection:
- EDA: 
- Preprocess: 
- Baselines: `CS5242_GroupID26.ipynb`  
    * MLP variants: MLPs, MLP, MLPv2 
    * Transformer  
    * BERT  
- Autogluon-Tabular: `run_autogluon.ipynb`  

## Project Set Up

This is a list of all requirements used in this project.

### Requirements for baselines

```
conda create -n CS5242-ABSC -y python=3.9 pip
conda activate CS5242
pip install jupyter
jupyter notebook
```

To install requirements, run `pip install -r requirements.txt`.

### Requirements for EDA images plot

```
pip install numpy pandas seaborn scipy plotly
pip install -U matplotlib
pip install -U kaleido
python eda_plot.py
jupyter notebook # aspects_fi.ipynb for feature importance analysis
```

## Framework Overview


## Note


## Acknowledgement
We are grateful to CS5242 for giving us such a valuable experience.  
Group26: Yan Zehong, Luo Jian, Zhao Yijing.  
