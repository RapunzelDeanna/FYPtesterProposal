# © 2025 Deanna White. All rights reserved.
#  COMP1682 Final Year Project
# Purpose: Regression prediction for box office revenue


import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Takes a while with cpi so just checking
start_time = time.time()

def load_and_prepare_data():
    """
        Load and prepare dataset for machine learning analysis.

        This function handles:
        - Dataset loading from CSV
        - Target column selection
        - Time-based data splitting
        - Feature scaling

        Returns:
            X_train_scaled, X_test_scaled, y_train, y_test, features, ds_name
    """
    ds_name = input("Enter the dataset name (without .csv): ")

    try:
        dataset = pd.read_csv(ds_name + '.csv')
        print(f"Dataset '{ds_name}' loaded successfully.")
        print(f"Shape: {dataset.shape}")

        # User selects target column
        print("\nAvailable columns:", dataset.columns.tolist())
        target_column = input("Enter the target column name: ")

        # Ensures the target_column is in the dataset
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
        train_data = dataset.iloc[:split_index]
        test_data = dataset.iloc[split_index:]
        # Checks what the features look like
        print("train data: ", train_data)
        print("test data: ", test_data)
        # For random split comparison
        # train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

        # Features selection
        features = [col for col in dataset.columns if col != target_column]
        X_train, y_train = train_data[features].values, train_data[target_column].values
        X_test, y_test = test_data[features].values, test_data[target_column].values
        print(f"Train Release Year Range: {train_data['release_year'].min()} - {train_data['release_year'].max()}")
        print(f"Test Release Year Range: {test_data['release_year'].min()} - {test_data['release_year'].max()}")

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, features, ds_name

    except FileNotFoundError:
        print(f"Error: The file {ds_name}.csv was not found.")
        return None, None, None, None, None, None


# Automatically select features and target column based on user input
def auto_select_features_target(dataset):
    """
        Automatically select features and target column based on user input.

        Parameters:
            dataset: Loaded pandas DataFrame

        Returns:
            tuple: (features, target_column) or (None, None) if selection fails
    """
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

# Hyperparameter Tuning
def tune_ridge(X_train, y_train):
    """
        Tune Ridge Regression hyperparameters using GridSearchCV.

        Parameters:
            X_train: Training features
            y_train: Training target

        Returns:
            Ridge: Best performing Ridge model
    """
    print("\nTuning Ridge Hyperparameters")
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
    model = Ridge(max_iter=10000)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Ridge Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_lasso(X_train, y_train):
    print("\nTuning Lasso Hyperparameters")
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
    model = Lasso(max_iter=10000)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Lasso Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_elasticnet(X_train, y_train):
    print("\nTuning ElasticNet Hyperparameters")
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1],
        'l1_ratio': [0.1, 0.5, 0.9]
    }
    model = ElasticNet()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best ElasticNet Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_decision_tree(X_train, y_train):
    print("\nTuning Decision Tree Hyperparameters")
    param_grid = {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    model = DecisionTreeRegressor(random_state=77)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Decision Tree Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_random_forest(X_train, y_train):
    print("\nTuning Random Forest Hyperparameters")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    }
    model = RandomForestRegressor(random_state=77)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Random Forest Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_extra_trees(X_train, y_train):
    print("\nTuning Extra Trees Hyperparameters")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    }
    model = ExtraTreesRegressor(random_state=77)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Extra Trees Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_gradient_boosting(X_train, y_train):
    print("\nTuning Gradient Boosting Hyperparameters")
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10]
    }
    model = GradientBoostingRegressor(random_state=77)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Gradient Boosting Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_adaboost(X_train, y_train):
    print("\nTuning AdaBoost Hyperparameters")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }
    model = AdaBoostRegressor(random_state=77)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best AdaBoost Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_xgboost(X_train, y_train):
    print("\nTuning XGBoost Hyperparameters")
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
    model = XGBRegressor(random_state=77, verbosity=0)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best XGBoost Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_lightgbm(X_train, y_train):
    print("\nTuning LightGBM Hyperparameters")
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 64]
    }
    model = LGBMRegressor(random_state=77, verbose=-1)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best LightGBM Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_catboost(X_train, y_train):
    print("\nTuning CatBoost Hyperparameters")
    param_grid = {
        'iterations': [100, 200],
        'learning_rate': [0.01, 0.1],
        'depth': [4, 6, 8]
    }
    model = CatBoostRegressor(random_state=77, verbose=0)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best CatBoost Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_svr(X_train, y_train):
    print("\nTuning SVR Hyperparameters (RBF vs Linear Kernels)")
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']  # 'gamma' is only relevant for 'rbf', will be ignored for 'linear'
    }
    model = SVR()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best SVR Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def tune_knn(X_train, y_train):
    print("\nTuning KNN Hyperparameters")
    param_grid = {
        'n_neighbors': [3, 5, 7, 9]
    }
    model = KNeighborsRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best KNN Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def get_models(X_train, y_train):
    """
        Initialize and tune multiple regression models.

        Returns:
            dict: Dictionary of tuned models with their names as keys
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": tune_ridge(X_train, y_train),
        "Lasso Regression": tune_lasso(X_train, y_train),
        "ElasticNet": tune_elasticnet(X_train, y_train),
        "Decision Tree": tune_decision_tree(X_train, y_train),
        "Random Forest": tune_random_forest(X_train, y_train),
        "Extra Trees": tune_extra_trees(X_train, y_train),
        "Gradient Boosting": tune_gradient_boosting(X_train, y_train),
        "AdaBoost": tune_adaboost(X_train, y_train),
        "XGBoost": tune_xgboost(X_train, y_train),
        "LightGBM": tune_lightgbm(X_train, y_train),
        "CatBoost": tune_catboost(X_train, y_train),
        "SVR": tune_svr(X_train, y_train),
        "KNN": tune_knn(X_train, y_train),
        }

    # Define base models for stacking
    base_models = [
         ("xgb", XGBRegressor(n_estimators=100, random_state=77)),
         ("lgbm", LGBMRegressor(n_estimators=100, random_state=77, verbose=0)),
         ("cat", CatBoostRegressor(iterations=100, random_state=77, verbose=0)),
         ("gbdt", GradientBoostingRegressor(n_estimators=100, random_state=77)),
         ("rf", RandomForestRegressor(n_estimators=100, random_state=77)),
         ("svr", SVR(kernel="rbf"))
    ]

    # Define the meta-model (final estimator)
    meta_model = Ridge(alpha=0.01, max_iter=10000)

    # Define the Stacking Ensemble
    stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

    # Add Stacking model to the list of models
    models["Stacking Ensemble"] = stacking_model

    return models

def validate_models(models, X_train, y_train, cv_splits=10):
    """
        Validate models using KFold cross-validation.

        Parameters:
            models: Dictionary of models to validate
            X_train: Training features
            y_train: Training target
            cv_splits: Number of cross-validation folds
    """
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=77)
        # Calculates mean for each model and displays
    for name, model in models.items():
        print(f"\nTraining {name} ...with cross validation")
        mse_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
        r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
        mse_mean = -mse_scores.mean()  # Convert negative MSE back to positive
        rmse_mean = np.sqrt(mse_mean)  # Compute RMSE


        print(f"\n{name} Validation:")
        print(f"  MSE: {mse_scores.mean():.2f}")
        print(f"  RMSE: {rmse_mean:.2f}")
        print(f"  R2: {r2_scores.mean():.2f}")


def test_models(ds_name, models, X_train, X_test, y_train, y_test, features):
    """
       Test models on the test set and generate visualizations.

       Parameters:
           ds_name: Dataset name for output files
           models: Dictionary of models to test
           X_train: Training features
           X_test: Test features
           y_train: Training target
           y_test: Test target
           features: List of feature names
    """
    # Store the results for plotting
    plot_data = []
    confusion_matrices = []
    feature_importance = []
    y_preds = {}
    for name, model in models.items():
        print(f"\nTesting {name} ...")

        # Train the model
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred_test = model.predict(X_test)
        y_preds[name] = y_pred_test
        plot_data.append((name, y_test, y_pred_test))

        # Calculate Metrics
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred_test)
        one_away_acc = one_away_accuracy(y_test, y_pred_test)
        exact_acc = exact_accuracy(y_test, y_pred_test)

        # Display results
        print(f"\n{name} Test Results:")
        print(f"  MSE: {mse:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R2: {r2:.2f}")
        print(f"  One-Away Accuracy: {one_away_acc:.2f}%")
        print(f"  Exact Accuracy: {exact_acc:.2f}%")


        # Collect Confusion Matrix data
        collect_confusion_matrices(name, y_test, y_pred_test, confusion_matrices)

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
    plot_confusion_matrices(confusion_matrices, ds_name)

    # Plot feature importance if any model supports it
    if feature_importance:
        plot_feature_importance(feature_importance, features, ds_name)

    return y_test, y_preds, plot_data


def important_features(ds_name, models, features, X_train, X_test, y_train, y_test, top_n=20):
    """
        Analyze feature importance across models and select top features.

        Parameters:
            ds_name: Dataset name for output files
            models: Dictionary of trained models
            features: List of feature names
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            top_n: Number of top features to select (default: 20)

        Returns:
            tuple: (X_train_selected, X_test_selected, selected_features, best_model_name)
    """
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

        # Special handling for stacking model
        elif isinstance(model, StackingRegressor):
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

    # Visualize the top N feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(selected_features[::-1], importance[top_features_idx][::-1], color="skyblue")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Most Important Features\n(Based on {best_model_name})")
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{ds_name}_important_feature_{best_model_name}.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    return X_train_selected, X_test_selected, selected_features, best_model_name


# Categorize the predicted and actual values into categories
def categorize_box_office(value):
    """
        Categorize box office performance into predefined categories.

        Parameters:
            value: float - Box office revenue value

        Returns:
            str: Category name ('Flop', 'Below Average', 'Average', 'Hit', or 'Blockbuster')
    """
    if value > 1:
        if value < 1e6:
            return "Flop"
        elif value < 5e6:
            return "Below Average"
        elif value < 40e6:
            return "Average"
        elif value < 150e6:
            return "Hit"
        else:
            return "Blockbuster"


# Calculate one-away accuracy
def one_away_accuracy(y_true, y_pred):
    """
        Calculate one-away accuracy for box office predictions.
        A prediction is considered "one-away" if it's either:
        1. Exactly correct
        2. One category higher or lower than the true category

        Parameters:
            y_true: array-like - True box office values
            y_pred: array-like - Predicted box office values

        Returns:
            float: Percentage of predictions that are either correct or one category away
    """
    true_categories = [categorize_box_office(value) for value in y_true]
    pred_categories = [categorize_box_office(value) for value in y_pred]

    category_map = {'Flop': 0, 'Below Average': 1, 'Average': 2, 'Hit': 3, 'Blockbuster': 4}

    correct = sum(
        abs(category_map[true] - category_map[pred]) <= 1
        for true, pred in zip(true_categories, pred_categories)
    )
    #Returns as percentage
    return correct / len(y_true) * 100


# Calculate exact accuracy
def exact_accuracy(y_true, y_pred):
    """
        Calculate exact accuracy for box office predictions.
        A prediction is considered correct only if it matches the true category exactly.

        Parameters:
            y_true: array-like - True box office values
            y_pred: array-like - Predicted box office values

        Returns:
            float: Percentage of exactly correct predictions
    """
    true_categories = [categorize_box_office(value) for value in y_true]
    pred_categories = [categorize_box_office(value) for value in y_pred]

    correct = sum(true == pred for true, pred in zip(true_categories, pred_categories))
    category_map = {'Flop': 0, 'Below Average': 1, 'Average': 2, 'Hit': 3, 'Blockbuster': 4}

    true_numeric = np.array([category_map[cat] for cat in true_categories])
    pred_numeric = np.array([category_map[cat] for cat in pred_categories])

    return correct / len(y_true) * 100



# Store confusion matrices for all models
def collect_confusion_matrices(name, y_true, y_pred, confusion_matrices):
    """
        Store confusion matrices for model evaluation.

        Parameters:
            name: str - Model name
            y_true: array-like - True box office values
            y_pred: array-like - Predicted box office values
            confusion_matrices: list - List to store (name, matrix) tuples
    """
    category_order = ['Flop', 'Below Average', 'Average', 'Hit', 'Blockbuster']
    category_map = {label: idx for idx, label in enumerate(category_order)}

    y_true_encoded = [category_map[label] for label in [categorize_box_office(value) for value in y_true]]
    y_pred_encoded = [category_map[label] for label in [categorize_box_office(value) for value in y_pred]]

    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
    confusion_matrices.append((name, cm))  # Store model name and its confusion matrix

def plot_confusion_matrices(confusion_matrices, ds_name):
    """
       Plot confusion matrices for multiple models in groups of 6.

       Parameters:
           confusion_matrices: list - List of (name, matrix) tuples
           ds_name: str - Dataset name for output files
    """
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
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=category_order)
            disp.plot(ax=axes[i], cmap="Blues", values_format="d")
            axes[i].set_title(name)

        # Hide any unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(f'{ds_name}_confusion_matrices_{start // models_per_figure}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()


def plot_feature_importance(feature_importance, feature_names, ds_name):
    """
       Plot feature importance for models that support it in a 3x2 grid across 3 PNG files.

       Parameters:
           feature_importance: list - List of (name, importance) tuples
           feature_names: list - List of feature names
           ds_name: str - Dataset name for output files

       Returns:
           dict: Dictionary mapping model names to their top features
    """
    selected_features_per_model = {}
    models_per_figure = 6  # Maximum number of models per figure (3x2 grid)
    num_models = len(feature_importance)

    # Calculate number of figures needed
    num_figures = -(-num_models // models_per_figure)  # Ceiling division

    # Create a figure for each set of 6 models
    for fig_num in range(num_figures):
        start = fig_num * models_per_figure
        end = min(start + models_per_figure, num_models)
        subset_importance = feature_importance[start:end]

        # Create figure with 3x2 grid
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        axes = axes.flatten()

        # Plot each model's feature importance
        for i, (name, importance) in enumerate(subset_importance):
            sorted_idx = np.argsort(importance)[::-1][:20]  # Sort by importance (top 20 features)
            top_importance = importance[sorted_idx]
            top_feature_names = np.array(feature_names)[sorted_idx]

            axes[i].barh(top_feature_names, top_importance, color="royalblue")
            axes[i].set_title(f"Feature Importance - {name}")
            axes[i].invert_yaxis()  # Ensure highest importance is at the top

            selected_features_per_model[name] = top_feature_names

        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Save the figure
        plt.tight_layout()
        plt.savefig(f'{ds_name}_feature_importance_{fig_num + 1}.png',
                    bbox_inches='tight', dpi=300)
        plt.close(fig)

    return selected_features_per_model


# Main program
def main():
    """
        Main execution function for the machine learning pipeline.

        This function orchestrates the entire workflow:
        1. Data loading and preparation
        2. Model training and validation
        3. Feature importance analysis
        4. Testing with selected features
    """
    X_train, X_test, y_train, y_test, features, ds_name = load_and_prepare_data()

    if X_train is None or X_test is None:
        print("Exiting due to data loading error.")
        return

    models = get_models(X_train, y_train)

    print("\nValidating models with cross-validation...")
    validate_models(models, X_train, y_train)

    print("\nTesting models on the test set...")
    test_models(ds_name, models, X_train, X_test, y_train, y_test, features)
    # Selecting important features
    X_train_selected, X_test_selected, selected_features, best_model_name = important_features(
        ds_name, models, features, X_train, X_test, y_train, y_test, top_n=20
    )

    # Print the selected features and the best model
    print(f"Best model: {best_model_name}")
    print(f"Selected top features: {selected_features}")
    # Re-test models with selected features
    print("\nTesting models on the selected features...")
    test_models(ds_name, models, X_train_selected, X_test_selected, y_train, y_test,
                                             selected_features)
    # End timer
    end_time = time.time()

    # Calculate time taken
    elapsed_time = end_time - start_time
    print("Time taken: ", elapsed_time)


if __name__ == "__main__":
    main()
