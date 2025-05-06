This project is for my Final Year Project. This dissertation explores the application of machine learning algorithms to predict movie box office revenue, a task with significant implications for the film industry. Both regression and classification approaches are studied, employing a range of models including linear regression, ensemble methods, and stacked ensemble. The study utilises three distinct datasets with varying movie features and release timelines to evaluate the generalisability and robustness of the predictive systems. Thorough preprocessing techniques are applied to each dataset, including feature engineering and addressing inflation. The model hyperparameters are tuned using Grid Search, and a 10-fold cross-validation strategy for model optimisation is used. Appropriate evaluation metrics are in place for both regression (\( R^2 \), MSE, RMSE) and classification (accuracy, precision, recall, F1-score). The one-away accuracy is measured on all models, allowing for comparisons. The findings offer insights into the most effective techniques for forecasting movie box office revenue under different data conditions. 

Preprocessed CSV files to use (regression):
dataset1.csv
dataset2.csv
dataset3.csv

Target Column: revenue


Preprocessed CSV files to use (Classification):
dataset1class.csv
dataset2class.csv
dataset3class.csv

Target Column: revenue_class

Run visual.py or Classification and input the preferred dataset and Target Column


Output (for regression models):
Training "Algorithm" with 10-Fold Cross-Validation...
"Model" Test Results:
  MSE (Mean): ######
  RMSE (Mean): ######
  R2 (Mean): ######
  One-Away Accuracy: ##.##%
  Exact Accuracy: ##.##%

Output (for classification models):
"Model" Test Results:
  Accuracy: #.###
  Precision: #.###
  Recall: #.###
  F1 Score: #.####
  One-Away Accuracy: ##.###



Expected vs actual movies in each categories sorted by algorithm models displayed at the end
