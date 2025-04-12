import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, \
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, RidgeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (mean_squared_error,r2_score,accuracy_score,f1_score,classification_report)

def load_and_prepare_data():
    ds_name = input("Enter the dataset name (without .csv): ")

    try:
        dataset = pd.read_csv(ds_name + '.csv')
        print(f"Dataset '{ds_name}' loaded successfully.")
        print(f"Shape: {dataset.shape}")

        # User selects target column
        print("\nAvailable columns:", dataset.columns.tolist())
        target_column = input("Enter the target column name: ")

        if target_column not in dataset.columns:
            print(f"Error: '{target_column}' not found in dataset.")
            return None, None, None, None, None

        # Ensure 'release_year' is in the dataset
        if 'release_year' not in dataset.columns:
            print("Error: 'release_year' column not found in dataset. Required for time-based splitting.")
            return None, None, None, None, None


        # Sort by release_year (ascending order)
        dataset = dataset.sort_values(by="release_year")

        # Split dataset (80% train, 20% test based on release_year)
        split_index = int(len(dataset) * 0.8)
        train_data = dataset.iloc[:split_index].copy()
        test_data = dataset.iloc[split_index:].copy()

        # Detect target type: numerical (regression) or categorical (classification)
        is_classification = dataset[target_column].dtype == 'object' or dataset[target_column].nunique() < 10

        # Define columns for log transformation
        log_columns = ['vote_count', 'popularity', 'budget', 'company_rev']

        # Apply log transformation
        for col in log_columns:
            if col in train_data.columns:  # Ensure the column exists
                train_data[col] = np.log1p(train_data[col])
                test_data[col] = np.log1p(test_data[col])

        # Feature selection
        features = [col for col in dataset.columns if col != target_column]
        X_train, y_train = train_data[features].values, train_data[target_column].values
        X_test, y_test = test_data[features].values, test_data[target_column].values

        # Validate target distribution
        validate_target_distribution(y_train, y_test)

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        print(f"Target type detected: {'Categorical' if is_classification else 'Continuous (regression)'}")
        return X_train, X_test, y_train, y_test, features, is_classification



    except FileNotFoundError:
        print(f"Error: The file {ds_name}.csv was not found.")
        return None, None, None, None, None


def validate_target_distribution(y_train, y_test):
    """Validate target distribution and detect potential issues"""
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)

    print("\nTarget Distribution:")
    print(f"Training classes found: {train_unique}")
    print(f"Training class counts: {dict(zip(train_unique, train_counts))}")
    print(f"Test classes found: {test_unique}")
    print(f"Test class counts: {dict(zip(test_unique, test_counts))}")

    # Check if all predictions are the same value
    if len(train_unique) == 1:
        raise ValueError("Warning: Training data contains only one class!")

    return train_unique, train_counts, test_unique, test_counts

# Automatically select features and target column based on user input
def auto_select_features_target(dataset):
    target_column = input("Enter the target column name: ").strip()

    # Check if target column exists in the dataset
    if target_column not in dataset.columns:
        print(f"Error: The target column '{target_column}' does not exist in the dataset.")
        return None, None

    # Select numeric columns as features
    features = dataset.select_dtypes(include=['number']).columns.tolist()

    # Remove target column from features if it's numeric
    if target_column in features:
        features.remove(target_column)

    # User is aware of the features being used
    print(f"Features selected: {features}")
    print(f"Target column selected: {target_column}")
    return features, target_column



def tune_ridge(X_train, y_train, classification=False):
    print("\nTuning Ridge Hyperparameters")
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
    if classification:
        model = RidgeClassifier()
        scoring = 'f1_weighted'
    else:
        model = Ridge(max_iter=10000)
        scoring = 'r2'
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Ridge Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_lasso(X_train, y_train, classification=False):
    print("\nTuning Lasso Hyperparameters")
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
    if classification:
        raise ValueError("Lasso doesn't support classification directly.")
    else:
        model = Lasso(max_iter=10000)
        scoring = 'r2'
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Lasso Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_elastic_net(X_train, y_train, classification=False):
    print("\nTuning ElasticNet Hyperparameters")
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1], 'l1_ratio': [0.2, 0.5, 0.8]}
    if classification:
        raise ValueError("ElasticNet doesn't support classification directly.")
    else:
        model = ElasticNet(max_iter=10000)
        scoring = 'r2'
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best ElasticNet Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_decision_tree(X_train, y_train, classification=False):
    print("\nTuning Decision Tree Hyperparameters")
    param_grid = {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]}
    if classification:
        model = DecisionTreeClassifier(random_state=77)
        scoring = 'f1_weighted'
    else:
        model = DecisionTreeRegressor(random_state=77)
        scoring = 'r2'
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Decision Tree Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_random_forest(X_train, y_train, classification=False):
    print("\nTuning Random Forest Hyperparameters")
    param_grid = {'n_estimators': [100, 300], 'max_depth': [10, 20], 'min_samples_split': [2, 5]}
    if classification:
        model = RandomForestClassifier(random_state=77)
        scoring = 'f1_weighted'
    else:
        model = RandomForestRegressor(random_state=77)
        scoring = 'r2'
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Random Forest Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_svr(X_train, y_train, classification=False):
    print("\nTuning SVR/SVC Hyperparameters")
    param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.1, 1], 'kernel': ['rbf']}
    if classification:
        model = SVC()
        scoring = 'f1_weighted'
    else:
        model = SVR()
        scoring = 'r2'
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best SVR/SVC Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def tune_xgboost(X_train, y_train, classification=False):
    print("\nTuning XGBoost Hyperparameters")

    # Define parameter grid for grid search
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [50, 100, 150],
        'gamma': [0, 0.1, 0.25]
    }

    if classification:
        # If classification, use XGBClassifier
        model = XGBClassifier(
            random_state=77,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0
        )
        scoring = 'f1_weighted'
    else:
        # If regression, use XGBRegressor
        model = XGBRegressor(
            random_state=77,
            verbosity=0
        )
        scoring = 'r2'

    # Initialize GridSearchCV with the model and parameter grid
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,  # Number of cross-validation folds
        scoring=scoring,
        n_jobs=-1,  # Parallelize the search
        verbose=0  # Set to 1 for detailed output
    )
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    print(f"Best XGBoost Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def tune_lightgbm(X_train, y_train, classification=False):
    print("\nTuning LightGBM Hyperparameters")
    param_grid = {'n_estimators': [100, 300], 'learning_rate': [0.01, 0.1], 'max_depth': [-1, 10]}
    if classification:
        model = LGBMClassifier(random_state=77)
        scoring = 'f1_weighted'
    else:
        model = LGBMRegressor(random_state=77, verbose=-1)
        scoring = 'r2'
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best LightGBM Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_catboost(X_train, y_train, classification=False):
    print("\nTuning CatBoost Hyperparameters")
    param_grid = {'iterations': [100, 300], 'learning_rate': [0.01, 0.1], 'depth': [4, 6]}
    if classification:
        model = CatBoostClassifier(random_state=77, verbose=0)
        scoring = 'f1_weighted'
    else:
        model = CatBoostRegressor(random_state=77, verbose=0)
        scoring = 'r2'
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best CatBoost Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_knn(X_train, y_train, classification=False):
    print("\nTuning KNN Hyperparameters")
    param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    if classification:
        model = KNeighborsClassifier()
        scoring = 'f1_weighted'
    else:
        model = KNeighborsRegressor()
        scoring = 'r2'
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best KNN Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_mlp(X_train, y_train, classification=False):
    print("\nTuning MLP Hyperparameters")
    param_grid = {
        'hidden_layer_sizes': [(64,), (64, 32)],
        'activation': ['relu'],
        'solver': ['adam'],
        'learning_rate_init': [0.0001, 0.001],
        'max_iter': [1000, 2000, 5000]
    }
    if classification:
        model = MLPClassifier(random_state=42)
        scoring = 'f1_weighted'
    else:
        model = MLPRegressor(random_state=42)
        scoring = 'r2'
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best MLP Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_



def get_models(X_train, y_train, is_classification):
    if is_classification:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Extra Trees": ExtraTreesClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "XGBoost": tune_xgboost(X_train, y_train),
            "LightGBM": tune_lightgbm(X_train, y_train),
            "CatBoost": tune_catboost(X_train, y_train),
            "SVC (RBF Kernel)": tune_svr(X_train, y_train),
            "KNN": tune_knn(X_train, y_train),
            "MLP (Neural Network)": tune_mlp(X_train, y_train)
        }

        base_models = [
            ("xgb", tune_xgboost(X_train, y_train)),
            ("lgbm", tune_lightgbm(X_train, y_train)),
            ("cat", tune_catboost(X_train, y_train)),
            ("gbdt", GradientBoostingClassifier()),
            ("rf", RandomForestClassifier()),
            ("svc", tune_svr(X_train, y_train))
        ]
        meta_model = LogisticRegression(max_iter=1000)

        stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            passthrough=False
        )
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": tune_ridge(X_train, y_train),
            "Lasso Regression": tune_lasso(X_train, y_train),
            "ElasticNet": tune_elastic_net(X_train, y_train),
            "Decision Tree": tune_decision_tree(X_train, y_train),
            "Random Forest": tune_random_forest(X_train, y_train),
            "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=77),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=77),
            "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=77),
            "XGBoost": tune_xgboost(X_train, y_train),
            "LightGBM": tune_lightgbm(X_train, y_train),
            "CatBoost": tune_catboost(X_train, y_train),
            "SVR (RBF Kernel)": tune_svr(X_train, y_train),
            "KNN": tune_knn(X_train, y_train),
            "MLP (Neural Network)": tune_mlp(X_train, y_train)
        }

        base_models = [
            ("xgb", tune_xgboost(X_train, y_train)),
            ("lgbm", tune_lightgbm(X_train, y_train)),
            ("cat", tune_catboost(X_train, y_train)),
            ("gbdt", GradientBoostingRegressor(n_estimators=100, random_state=77)),
            ("rf", tune_random_forest(X_train, y_train)),
            ("svr", tune_svr(X_train, y_train))
        ]
        meta_model = Ridge(alpha=0.01, max_iter=10000)

        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            passthrough=False
        )

    models["Stacking Ensemble"] = stacking_model

    return models


def validate_models(is_classification, models, X_train, y_train, cv_splits=10):
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=77)
        # Calculates mean for each model and displays
    for name, model in models.items():
        print(f"\nTraining {name} ...with cross validation")


        if is_classification:
            # For classification targets
            accuracy = []
            f1 = []

            for train_idx, val_idx in kf.split(X_train):
                X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                model.fit(X_train_fold, y_train_fold)

                # Get predictions
                y_pred = model.predict(X_val_fold)

                # Ensure predictions are class labels
                if hasattr(model, 'predict_proba'):
                    y_pred = model.predict(X_val_fold)
                else:
                    # For models that output continuous values
                    y_pred = np.round(y_pred).astype(int)
                    y_pred = np.clip(y_pred, 1, 5)  # Ensure predictions are in [1,5]

                # Calculate metrics for this fold
                acc = accuracy_score(y_val_fold, y_pred)
                f1_score_val = f1_score(y_val_fold, y_pred, average='macro')

                accuracy.append(acc)
                f1.append(f1_score_val)

            print(f"\n{name} Validation:")
            print(f"  Accuracy: {np.mean(accuracy):.2f}")
            print(f"  F1 Score: {np.mean(f1):.2f}")
        else:
            mse_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
            r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
            mse_mean = -mse_scores.mean()
            rmse_mean = np.sqrt(mse_mean)

            print(f"\n{name} Validation:")
            print(f"  MSE: {mse_mean:.2f}")
            print(f"  RMSE: {rmse_mean:.2f}")
            print(f"  R2: {r2_scores.mean():.2f}")


def test_models(models, X_train, X_test, y_train, y_test, features, is_classification):
    # Store the results for plotting
    plot_data = []
    confusion_matrices = []
    feature_importance = []
    y_preds = {}

    validate_target_distribution(y_train, y_test)

    for name, model in models.items():
        print(f"\nTesting {name} ...")

        # Train the model
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred_test = model.predict(X_test)
        y_preds[name] = y_pred_test
        plot_data.append((name, y_test, y_pred_test))

        if is_classification:
            # For classification: ensure predictions are discrete class labels
            if hasattr(model, 'predict_proba'):
                # If the model has `predict_proba`, use the class probabilities to get class labels
                y_pred_class = np.argmax(model.predict_proba(X_test), axis=1)
            else:
                # If it's a classification model without `predict_proba`, just round continuous outputs to the nearest class
                y_pred_class = np.round(y_pred_test).astype(int)
                y_pred_class = np.clip(y_pred_class, 1, 5)  # Ensure predictions are within class bounds [1,5]

            # Now calculate classification metrics
            acc = accuracy_score(y_test, y_pred_class)
            f1 = f1_score(y_test, y_pred_class, average='weighted')
            print(f"\n{name} Test Results:")
            print(f"  Accuracy: {acc:.2f}")
            print(f"  F1 Score: {f1:.2f}")
            print(f"\nClassification Report:\n{classification_report(y_test, y_pred_class)}")

            # Collect Confusion Matrix
            collect_confusion_matrices(name, y_test, y_pred_class, confusion_matrices, is_classification)

        else:
            # For regression: calculate regression metrics
            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred_test)
            one_away_acc = one_away_accuracy(y_test, y_pred_test, is_classification)
            exact_acc = exact_accuracy(y_test, y_pred_test, is_classification)

            print(f"\n{name} Test Results:")
            print(f"  MSE: {mse:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  R2: {r2:.2f}")
            print(f"  One-Away Accuracy: {one_away_acc:.2f}%")
            print(f"  Exact Accuracy: {exact_acc:.2f}%")

            # Collect Confusion Matrix (for regression, treat it as a continuous prediction)
            collect_confusion_matrices(name, y_test, y_pred_test, confusion_matrices, is_classification)

        if hasattr(model, "feature_importances_"):  # Tree-based models
            importance = model.feature_importances_
            feature_importance.append((name, importance))

        elif hasattr(model, "coef_"):  # Linear models
            importance = np.abs(model.coef_)
            if importance.ndim > 1:  # Handle multi-output case
                importance = importance.mean(axis=0)
            feature_importance.append((name, importance))

        else:
            print(f"{name} does not have feature importance or coefficients.")

    # After all models have been evaluated, plot confusion matrices
    plot_confusion_matrices(confusion_matrices)

    # Plot feature importance if any model supports it
    if feature_importance:
        plot_feature_importance(feature_importance, features)

    return y_test, y_preds, plot_data  # Return test results


def important_features(models, features, X_train, X_test, y_train, y_test, top_n=20):
    feature_importance = {}
    model_performance = {}

    # Evaluate each model's performance and get feature importances
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred_test = model.predict(X_test)

        # Calculate R² score to evaluate model performance
        r2 = r2_score(y_test, y_pred_test)

        # Store performance metrics for model comparison
        model_performance[name] = r2

        if hasattr(model, "feature_importances_"):
            feature_importance[name] = model.feature_importances_
        elif hasattr(model, "coef_"):  # For linear models
            importance = np.abs(model.coef_)
            if importance.ndim > 1:  # Handle multi-output case
                importance = importance.mean(axis=0)
            feature_importance[name] = importance

        elif isinstance(model, StackingRegressor):  # Special handling for stacking model
            stacked_importance = []
            for estimator in model.estimators_:
                if isinstance(estimator, tuple):  # Unpack if named tuple
                    base_name, base_model = estimator
                else:  # If it's just the model object
                    base_name, base_model = str(type(estimator).__name__), estimator

                if hasattr(base_model, "feature_importances_"):
                    stacked_importance.append(base_model.feature_importances_)
                elif hasattr(base_model, "coef_"):
                    stacked_importance.append(np.abs(base_model.coef_))

            if stacked_importance:
                # Compute mean feature importance across base models
                feature_importance[name] = np.mean(stacked_importance, axis=0)

    # Choose the best model based on R²
    best_model_name = max(model_performance, key=model_performance.get)  # model with highest R²

    print(f"Best model based on R²: {best_model_name} with R²: {model_performance[best_model_name]:.2f}")

    # Get the feature importances from the best model
    importance = feature_importance[best_model_name]

    # Sort the features based on importance (descending order)
    sorted_idx = np.argsort(importance)[::-1]
    top_features_idx = sorted_idx[:top_n]  # Select the top N features

    # Get the top N features' names
    selected_features = np.array(features)[top_features_idx]

    # Filter the dataset to include only the top features
    X_train_selected = X_train[:, top_features_idx]
    X_test_selected = X_test[:, top_features_idx]

    return X_train_selected, X_test_selected, selected_features, best_model_name


def categorize_box_office(value, is_classification):
    if is_classification:
        category_map = {
            1: "Flop", 2: "Below Average",
            3: "Average", 4: "Hit", 5: "Blockbuster"
        }
        category = category_map.get(value, "Unknown")
    else:
        if value < 0:
            category = "Unknown"
        elif value < 5e6:
            category = "Flop"
        elif value < 15e6:
            category = "Below Average"
        elif value < 30e6:
            category = "Average"
        elif value < 60e6:
            category = "Hit"
        else:
            category = "Blockbuster"

    if category == "Unknown":
        print(f"[Warning] Unknown category for value: {value} (is_classification={is_classification})")

    return category


def debug_predictions(y_true, y_pred, is_classification):
    """
    Debug function to track down negative values in predictions
    """
    print("\nDebug Analysis:")
    print("-" * 50)

    # Check for negative values in predictions
    negative_predictions = [(i, pred) for i, pred in enumerate(y_pred) if pred < 0]

    if negative_predictions:
        print(f"Found {len(negative_predictions)} negative predictions:")
        for idx, value in negative_predictions[:5]:  # Show first 5 examples
            true_value = y_true[idx]
            category = categorize_box_office(value, is_classification)
            print(f"Index {idx}:")
            print(f"  Prediction: {value:.2f}")
            print(f"  True value: {true_value:.2f}")
            print(f"  Category: {category}")
            print("-" * 30)
    else:
        print("No negative values found in predictions")

    # Check for negative values in true values
    negative_true = [(i, true) for i, true in enumerate(y_true) if true < 0]
    if negative_true:
        print(f"\nFound {len(negative_true)} negative true values:")
        for idx, value in negative_true[:5]:
            print(f"Index {idx}: {value:.2f}")
    else:
        print("\nNo negative values found in true values")
# Calculate one-away accuracy
def one_away_accuracy(y_true, y_pred, is_classification):
    # Categorize both true and predicted values
    debug_predictions(y_true, y_pred, is_classification)
    true_categories = [categorize_box_office(value, is_classification) for value in y_true]
    pred_categories = [categorize_box_office(value, is_classification) for value in y_pred]

    # Define category mapping for numerical comparison
    category_map = {'Flop': 0, 'Below Average': 1, 'Average': 2, 'Hit': 3, 'Blockbuster': 4}

    # Calculate one-away accuracy
    total_pairs = len(y_true)
    correct_predictions = sum(
        abs(category_map[true] - category_map[pred]) <= 1
        for true, pred in zip(true_categories, pred_categories)
        if true != "Unknown" and pred != "Unknown"
    )

    # Return percentage of accurate predictions
    return (correct_predictions / total_pairs * 100) if total_pairs > 0 else 0




def exact_accuracy(y_true, y_pred, is_classification):
    true_categories = [categorize_box_office(value, is_classification) for value in y_true]
    pred_categories = [categorize_box_office(value, is_classification) for value in y_pred]

    correct = sum(true == pred for true, pred in zip(true_categories, pred_categories))

    return correct / len(y_true) * 100  # Return percentage of exact accuracy



def collect_confusion_matrices(name, y_true, y_pred, confusion_matrices, is_classification):
    # Convert to categorical labels appropriately
    y_true_categories = [categorize_box_office(value, is_classification) for value in y_true]
    y_pred_categories = [categorize_box_office(value, is_classification) for value in y_pred]

    # Create confusion matrix with explicit labels
    cm = confusion_matrix(
        y_true_categories,
        y_pred_categories,
        labels=["Flop", "Below Average", "Average", "Hit", "Blockbuster"]
    )

    unique_true = len(set(y_true_categories))
    unique_pred = len(set(y_pred_categories))

    print(f"\n{name} Class Distribution:")
    print(f"True classes found: {sorted(set(y_true_categories))}")
    print(f"Predicted classes found: {sorted(set(y_pred_categories))}")

    if unique_true > 1 and unique_pred > 1:
        confusion_matrices.append((name, cm))
    else:
        print(f"Skipping {name}:")
        print(f"- Number of true classes: {unique_true}")
        print(f"- Number of predicted classes: {unique_pred}")

    print("\nDebug Information:")
    print(f"Unique true categories: {sorted(set(y_true_categories))}")
    print(f"Unique predicted categories: {sorted(set(y_pred_categories))}")
    print("\nFirst few examples:")
    for i in range(min(5, len(y_true_categories))):
        print(f"Example {i + 1}: Actual={y_true_categories[i]}, Predicted={y_pred_categories[i]}")

def plot_confusion_matrices(confusion_matrices):
    num_models = len(confusion_matrices)
    models_per_figure = 6  # Each figure should contain at most 6 confusion matrices
    cols = 3  # 3 columns per row

    for start in range(0, num_models, models_per_figure):
        end = min(start + models_per_figure, num_models)
        subset_matrices = confusion_matrices[start:end]

        rows = -(-len(subset_matrices) // cols)  # Ceiling division
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
        axes = axes.flatten()

        category_order = ['Flop', 'Below Average', 'Average', 'Hit', 'Blockbuster']

        for i, (name, cm) in enumerate(subset_matrices):
            # Print the actual values in the confusion matrix
            print(f"\nConfusion Matrix for {name}:")
            print(cm)

            # Ensure confusion matrix has all categories
            cm_full = np.zeros((len(category_order), len(category_order)))
            cm_full[:cm.shape[0], :cm.shape[1]] = cm

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm_full,
                display_labels=category_order
            )
            disp.plot(ax=axes[i], cmap="Blues", values_format=".2g")
            axes[i].set_title(name)

        # Hide any unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


def plot_feature_importance(feature_importance, feature_names):
    """Plot feature importance for models that support it in a 3x2 grid (6 models per figure)."""
    selected_features_per_model = {}
    models_per_figure = 6  # Maximum number of models per figure
    num_models = len(feature_importance)

    for start in range(0, num_models, models_per_figure):
        end = min(start + models_per_figure, num_models)
        subset_importance = feature_importance[start:end]

        cols = 3  # Number of columns per row
        rows = -(-len(subset_importance) // cols)  # Ceiling division to get row count
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))  # Adjust figure size
        axes = axes.flatten()  # Flatten axes for easy iteration

        for i, (name, importance) in enumerate(subset_importance):
            sorted_idx = np.argsort(importance)[::-1][:20]  # Sort by importance (top 20 features)
            top_importance = importance[sorted_idx]
            top_feature_names = np.array(feature_names)[sorted_idx]

            axes[i].barh(top_feature_names, top_importance, color="royalblue")
            axes[i].set_title(f"Feature Importance - {name}")
            axes[i].invert_yaxis()  # Ensure highest importance is at the top

            selected_features_per_model[name] = top_feature_names
        # Hide unused subplots in the last figure
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

        return selected_features_per_model


# Main program
def main():
    X_train, X_test, y_train, y_test, features, is_classification = load_and_prepare_data()

    if X_train is None or X_test is None:
        print("Exiting due to data loading error.")
        return

    models = get_models(X_train, y_train, is_classification)

    print("\nValidating models with cross-validation...")
    validate_models(is_classification, models, X_train, y_train)


    print("\nTesting models on the test set...")
    test_models(models, X_train, X_test, y_train, y_test, features, is_classification)
    # Selecting important features
    X_train_selected, X_test_selected, selected_features, best_model_name = important_features(
        models, features, X_train, X_test, y_train, y_test, top_n=20
    )

    # Print the selected features and the best model
    print(f"Best model: {best_model_name}")
    print(f"Selected top features: {selected_features}")
    # Re-test models with selected features
    print("\nTesting models on the selected features...")
    y_test, y_preds, plot_data = test_models(models, X_train_selected, X_test_selected, y_train, y_test,
                                             selected_features, is_classification)



if __name__ == "__main__":
    main()
