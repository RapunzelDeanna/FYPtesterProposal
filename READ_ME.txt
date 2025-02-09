Preprocessed CSV files to use:
prep_movies.csv (Target Column: AdjBoxOffice)

TMBD2Norm.csv (Target Column: revenue_adj)(no keywords, runs faster and more reliable atm)
or df.csv (Target Column: revenue_adj)(includes Keywords)

Movies1M.csv is a work in progress (Target column: revenue)
Movies1Msample.csv

Run Regression.py and input the preferred dataset and Target Column


Output:
Training "Algorithm" with 10-Fold Cross-Validation...
Linear Regression:
  MSE (Mean): ######
  RMSE (Mean): ######
  R2 (Mean): ######
  One-Away Accuracy: ##.##%
  Exact Accuracy: ##.##%

Expected vs actual movies in each categories sorted by algorithm models displayed at the end