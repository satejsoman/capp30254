# hw5 - further pipeline improvement 

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com) [![forthebadge](https://forthebadge.com/images/badges/gluten-free.svg)](https://forthebadge.com) [![forthebadge](https://forthebadge.com/images/badges/as-seen-on-tv.svg)](https://forthebadge.com)

## code structure

The `pipeline` library implementation and unit tests are in the `code/pipeline` directory. The sample application, a DonorsChoose project funding predictor, is located at `code/donors_choose.py`. Configuration options for this pipeline are set in `code/config.yml`.

Instructions for running the pipeline are listed below. 

## running the pipeline
The following commands should be run in the `hw5` directory: 

### 1/ set up a virtual environment 
```
python3 -mvenv .hw5
. .hw5/bin/activate 
```

### 2/ install requirements
```
pip3 install -r code/requirements.txt
```

### 3/ execute DonorsChoose classification pipeline
```
python code/donors_choose.py
```
## other possible actions: 
### run unit tests and end-to-end smoketest
```
pytest
```

### view PR curves for latest pipeline: 
```
ls code/output/*-LATEST/*.png
```

## analysis
please see the `latex/mlpp-hw5.pdf` file for analysis and results 
