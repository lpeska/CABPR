# CABPR
Content-Alignments for BPR


Python library for Content_Aligned Bayesian Personalized Ranking Matrix Factorization 
--------

version 1.0, January 31, 2018

--------
This package is written by:

Ladislav Peska,

Dept. of Software Engineering, Charles University in Prague, Czech Republic

Email: peska@ksi.mff.cuni.cz

Furthre information can be found on:
http://www.ksi.mff.cuni.cz/~peska/CABPR

-------
Some functions used in this package are based on the PyDTI package by Yong Liu,
https://github.com/stephenliu0423/PyDTI

This package also uses Rank metrics implementation by Brandyn White (included as rank_metrics.py)

--------
CABPR works on Python 3.6.
--------
CABPR requires NumPy, scikit-learn, SciPy and TensorFlow to run.
To get the results of different methods, please run cabpr_sparse.py. The __main__ part of the cabpr_sparse.py runs monte-carlo CV on extended MovieLens1M dataset or extended LOD-RecSys with internal hyperparameter tuning. The datasets are available on http://www.ksi.mff.cuni.cz/~peska/CABPR. However, CABPR method can input any binary user preference matrix and arbitrary many user and object-based similarity matrices.
