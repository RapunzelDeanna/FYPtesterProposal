Preprocessed CSV files to use:
movies_inflation+director.csv (Target Column: AdjBoxOffice)
TMBD2Norm.csv or df.csv (Target Column: revenue_adj)
Movies1M.csv is a work in progress (Target column: revenue)


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