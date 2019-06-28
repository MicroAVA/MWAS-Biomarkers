# MWAS-Biomarkers

This repo contains the code to reproduce all of the analyses in "Robust biomarker discovery for micorbiome-wide association stuties", Qiang Zhu et al. 2019 (https://doi.org/10.1016/j.ymeth.2019.06.012).

This work is based on Deep Forest: (https://arxiv.org/abs/1702.08835)

The data is available on MetAML: (http://dx.plos.org/10.1371/journal.pcbi.1004977)


# Reproducing analyses
If you want to get the feature selection result, you can run
```
feature_selection.py
```
then there will be a file under the output directory.

If you want to reproduce the evaluation, please run 
```
plot_auc_curve.py
```
If you want to calculate the Kuncheva index (https://dl.acm.org/citation.cfm?id=1295370), please run
```
calculate_kuncheva_index.py
```





## Installing

To re-make all of the analyses, you'll first need to install the required
modules.

You should probably do this in a Python 3 virtual environment.

```
conda create -n MWAS-Biomarkers python=3.6
source activate MWAS-Biomarkers
conda install pip
pip install -r requirements.txt
```

#### data

All data-related files are (or will be) in `lib/gcforest/data/`:


#### source code

All of the code is in the `lib/` folder:

* `gcforest`: the implementation of Deep Forest
* `output`: output for feature selection etc
* `util`: various functions and modules used in other scripts

