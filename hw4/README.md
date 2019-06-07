# hw4 - clustering!
![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg) ![forthebadge](https://forthebadge.com/images/badges/approved-by-george-costanza.svg) ![forthebadge](https://forthebadge.com/images/badges/does-not-contain-treenuts.svg)
## code structure
The `pipeline` library implementation and unit tests are in the `code/pipeline` directory. The sample application, a DonorsChoose project funding predictor, is located at `code/donors_choose.py`. Configuration options for this pipeline are set in `code/config.yml`.
## how to run this notebook:
The following commands should be run in the `hw4` directory: 

### 1/ set up a virtual environment 
```
python3 -mvenv .venv
. .venv/bin/activate 
```

### 2/ install requirements
```
pip3 install -r code/requirements.txt
```

### 4/ install ipython kernel in virtual environment
```
ipython kernel install --user --name=.venv
```

### 4/ execute notebook in browser 
navigate to the `donors_choose_clustering.ipynb` from the landing page at `localhost:8888`

alternatively, the notebook can be viewed statically on [Github](https://github.com/satejsoman/capp30254/blob/master/hw4/code/donors_choose_clustering.ipynb) or [nbviewer](https://nbviewer.jupyter.org/github/satejsoman/capp30254/blob/master/hw4/code/donors_choose_clustering.ipynb)

## analysis
please see the `latex/mlpp-hw4.pdf` file for analysis and results 
