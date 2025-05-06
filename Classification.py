# © 2025 Deanna White. All rights reserved.
#  COMP1682 Final Year Project
# Purpose: Classification prediction for box office revenue

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, \
    precision_score, recall_score, f1_score
from sklearn.ensemble import StackingClassifier
import time

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

        if target_column not in dataset.columns:
            print(f"Error: '{target_column}' not found in dataset.")
            return None, None, None, None, None, None

        # Ensure 'release_year' is in the dataset
        if 'release_year' not in dataset.columns:
            print("Error: 'release_year' column not found in dataset. Required for time-based splitting.")
            return None, None, None, None, None, None

        # Sort by release_year (ascending order)
        dataset = dataset.sort_values(by="release_year")
        # Split dataset (80% train, 20% test based on release_year)
        split_index = int(len(dataset) * 0.8)
        train_data = dataset.iloc[:split_index]
        test_data = dataset.iloc[split_index:]
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


def get_models(X_train, y_train):
    """
        Initialize and tune multiple machine learning models.

        Returns:
            dict: Dictionary of tuned models with their names as keys
        """
    models = {
        "Logistic Regression": tune_logistic_regression(X_train, y_train),
        "Ridge Classifier": tune_ridge_classifier(X_train, y_train),
        "Decision Tree": tune_decision_tree(X_train, y_train),
        "Random Forest": tune_random_forest(X_train, y_train),
        "Gradient Boosting": tune_gradient_boosting(X_train, y_train),
        "XGBoost": tune_xgboost(X_train, y_train),
        "LightGBM": tune_lightgbm(X_train, y_train),
        "CatBoost": tune_catboost(X_train, y_train),
        "SVM": tune_svm(X_train, y_train),
        "KNN": tune_knn(X_train, y_train),
        "MLP": tune_mlp(X_train, y_train),
    }

    # Define base models for stacking
    base_models = [
        ("xgb", tune_xgboost(X_train, y_train)),
        ("lgbm", tune_lightgbm(X_train, y_train)),
        ("cat", tune_catboost(X_train, y_train)),
        ("gbdt", tune_gradient_boosting(X_train, y_train)),
        ("rf", tune_random_forest(X_train, y_train)),
        ("svm", tune_svm(X_train, y_train))
    ]

    # Define the meta-model (final estimator)
    meta_model = LogisticRegression(random_state=77)

    # Define the Stacking Ensemble
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

    # Add Stacking model to the list of models
    models["Stacking Ensemble"] = stacking_model

    return models


def tune_logistic_regression(X_train, y_train):
    """
        Tune Logistic Regression hyperparameters using GridSearchCV.

        Parameters:
            X_train: Training features
            y_train: Training target

        Returns:
            Best performing Logistic Regression model
        """
    print("\nTuning Logistic Regression Hyperparameters")
    # Define parameter grids for different solver/penalty combinations
    param_grids = [
        {
            'solver': ['saga'],  # saga supports all penalties including 'none'
            'penalty': ['elasticnet', 'l1', 'l2', None],  # Use None instead of 'none'
            'max_iter': [1000, 2000, 3000],
            'C': [0.1, 1, 10]
        },
        {
            'solver': ['newton-cg', 'lbfgs'],  # these only support 'l2' and None
            'penalty': ['l2', None],
            'max_iter': [1000, 2000, 3000],
            'C': [0.1, 1, 10]
        }
    ]
    model = LogisticRegression(random_state=77)
    grid_search = GridSearchCV(model, param_grids, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Logistic Regression Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def tune_ridge_classifier(X_train, y_train):
    print("\nTuning Ridge Classifier Hyperparameters")
    param_grid = {
        'alpha': [0.1, 1, 10],
        'max_iter': [1000, 2000, 3000]
    }
    model = RidgeClassifier(random_state=77)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Ridge Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def tune_decision_tree(X_train, y_train):
    print("\nTuning Decision Tree Hyperparameters")
    param_grid = {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10]
    }
    model = DecisionTreeClassifier(random_state=77)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Decision Tree Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def tune_random_forest(X_train, y_train):
    print("\nTuning Random Forest Hyperparameters")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    model = RandomForestClassifier(random_state=77)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Random Forest Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def tune_gradient_boosting(X_train, y_train):
    print("\nTuning Gradient Boosting Hyperparameters")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10]
    }
    model = GradientBoostingClassifier(random_state=77)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Gradient Boosting Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def tune_xgboost(X_train, y_train):
    print("\nTuning XGBoost Hyperparameters")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10]
    }
    model = XGBClassifier(random_state=77, verbosity=0)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best XGBoost Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def tune_lightgbm(X_train, y_train):
    print("\nTuning LightGBM Hyperparameters")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 62, 127]
    }
    model = LGBMClassifier(random_state=77, verbose=-1)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best LightGBM Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def tune_catboost(X_train, y_train):
    print("\nTuning CatBoost Hyperparameters")
    param_grid = {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [3, 5, 10]
    }
    model = CatBoostClassifier(random_state=77, verbose=0)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best CatBoost Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def tune_svm(X_train, y_train):
    print("\nTuning SVM Hyperparameters")
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'probability': [True]
    }
    model = SVC()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best SVM Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def tune_knn(X_train, y_train):
    print("\nTuning KNN Hyperparameters")
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance']
    }
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best KNN Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def tune_mlp(X_train, y_train):
    print("\nTuning MLP Hyperparameters")
    param_grid = {
        'hidden_layer_sizes': [(50, 50), (100, 100), (50, 100, 50)],
        'max_iter': [1000, 2000, 3000],
        'alpha': [0.0001, 0.001, 0.01]
    }
    model = MLPClassifier(random_state=77)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best MLP Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def one_away_accuracy(y_true, y_pred):
    """
    Calculate one-away accuracy for box office predictions.
    Returns percentage of predictions that are either correct or off by one category.
    """
    correct = sum(
        abs(true - pred) <= 1
        for true, pred in zip(y_true, y_pred)
    )
    return float(correct / len(y_true) * 100)

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

    for name, model in models.items():
        print(f"\nTraining {name} ...with cross validation")
        accuracy_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
        print(f"{name} Cross-Validation Accuracy: {accuracy_scores.mean():.3f} (+/- {accuracy_scores.std() * 2:.3f})")


def test_models(ds_name, models, X_train, X_test, y_train, y_test, features, selected_features=None):
    """
        Test models on the test set and generate visualizations.

        Parameters:
            ds_name: Dataset name
            models: Dictionary of models to test
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            features: List of feature names
            selected_features: Optional list of selected features to use
        """
    confusion_matrices = []
    category_comparisons = []
    feature_importances = []  # Store feature importance data

    # Process each model
    for idx, (name, model) in enumerate(models.items()):
        print(f"\nTesting {name} ...")

        # Select features if specified
        if selected_features is None:
            X_train_used = X_train
            X_test_used = X_test
        else:
            X_train_used = X_train[:, selected_features]
            X_test_used = X_test[:, selected_features]
            print(f"Using {len(selected_features)} selected features")

        # Train model
        model.fit(X_train_used, y_train)

        # Predict and calculate metrics
        y_pred_test = model.predict(X_test_used)
        accuracy = accuracy_score(y_test, y_pred_test)

        # Calculate additional metrics
        precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        one_away = one_away_accuracy(y_test, y_pred_test)

        print(f"\n{name} Test Results:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        print(f"  One-Away Accuracy: {one_away:.3f}")

        # Store confusion matrix and category comparison data
        cm = confusion_matrix(y_test, y_pred_test)
        confusion_matrices.append((name, cm, np.unique(y_test)))
        category_comparisons.append((name, y_test, y_pred_test))

        # Store feature importance data instead of plotting immediately
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = np.abs(model.coef_)
            if coef.ndim == 1:  # Single coefficient per feature
                importance = coef
            else:  # Multiple coefficients (e.g., multi-class)
                importance = coef.mean(axis=0)
        else:
            importance = None

        feature_importances.append((name, importance))

    # Plot confusion matrices in groups of 6
    plot_confusion_matrices(confusion_matrices, ds_name)

    # Plot category comparisons in groups of 6
    num_models = len(category_comparisons)
    models_per_figure = 6

    for start in range(0, num_models, models_per_figure):
        end = min(start + models_per_figure, num_models)
        subset_comparisons = category_comparisons[start:end]

        fig, axes = plt.subplots(
            nrows=-(-len(subset_comparisons) // 3),
            ncols=3,
            figsize=(18, 4 * (-(-len(subset_comparisons) // 3)))
        )

        axes = np.array([axes]) if len(axes.shape) == 1 else axes

        for i, (name, y_true, y_pred) in enumerate(subset_comparisons):
            ax = axes[i // 3, i % 3]
            plot_category_comparison(ds_name, ax, y_true, y_pred)
            ax.set_title(name)

        fig.tight_layout()
        plt.savefig(f'{ds_name}_category_comparison_group_{start}split.png', bbox_inches='tight', dpi=300)
        plt.show()


    # Plot feature importance in groups of 6
    plot_feature_importances(feature_importances, features, ds_name)


def plot_confusion_matrices(confusion_matrices, ds_name):
    """
        Plot confusion matrices for multiple models in groups of 6.

        Parameters:
            confusion_matrices: List of (name, matrix, labels) tuples
            ds_name: Dataset name for filename
        """
    num_models = len(confusion_matrices)
    models_per_figure = 6
    cols = 3

    for start in range(0, num_models, models_per_figure):
        end = min(start + models_per_figure, num_models)
        subset_matrices = confusion_matrices[start:end]

        rows = -(-len(subset_matrices) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
        axes = axes.flatten()

        for i, (name, cm, labels) in enumerate(subset_matrices):
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=labels
            )
            disp.plot(ax=axes[i], cmap="Blues", values_format="d")
            axes[i].set_title(name)

        # Hide any unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(f'{ds_name}_confusion_matrix_{start // models_per_figure}split.png',
                    bbox_inches='tight', dpi=300)
        plt.close(fig)


def plot_category_comparison(ds_name, ax, y_true, y_pred, classes=None):
    """
    Plot comparison between true and predicted classifications.

    Parameters:
        ds_name: Dataset name
        ax: Matplotlib axis to plot on
        y_true: True labels
        y_pred: Predicted labels
        classes: Optional list of class labels
    """
    if classes is None:
        classes = np.arange(5)

    # Ensure 1D arrays for pandas operations
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    # Count actual occurrences in each class
    true_counts = pd.Series(y_true).value_counts().reindex(classes, fill_value=0)

    # Count predictions made for each class
    pred_counts = pd.Series(y_pred).value_counts().reindex(classes, fill_value=0)

    # Count correctly classified instances per class
    correct_mask = y_true == y_pred
    correct_counts = pd.Series(y_true[correct_mask]).value_counts().reindex(classes, fill_value=0)

    # Calculate incorrect classifications
    incorrect_counts = pred_counts - correct_counts

    x = np.arange(len(classes))
    width = 0.35

    # Plot true distribution
    ax.bar(x - width / 2, true_counts, width=width, color='blue', alpha=0.6, label='Actual Distribution')

    # Plot predictions stacked
    ax.bar(x + width / 2, correct_counts, width=width, color='green', label='Correct Predictions')
    ax.bar(x + width / 2, incorrect_counts, width=width, bottom=correct_counts, color='red', alpha=0.6,
           label='Incorrect Predictions')

    ax.set_title('Actual vs Predicted Class Distribution')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(loc='upper right')
    plt.tight_layout()

def plot_feature_importances(feature_importances, features, ds_name):
    """
    Plot feature importance scores for multiple models in groups of 6.

    Parameters:
        feature_importances: List of (name, importance) tuples
        features: List of feature names
        ds_name: Dataset name for filename
    """
    num_models = len(feature_importances)
    models_per_figure = 6

    for start in range(0, num_models, models_per_figure):
        end = min(start + models_per_figure, num_models)
        subset_importances = feature_importances[start:end]

        fig, axes = plt.subplots(
            nrows=-(-len(subset_importances) // 3),
            ncols=3,
            figsize=(18, 4 * (-(-len(subset_importances) // 3)))
        )

        axes = np.array([axes]) if len(axes.shape) == 1 else axes

        for i, (name, importance) in enumerate(subset_importances):
            ax = axes[i // 3, i % 3]

            if importance is not None:
                # Get top 20 features
                sorted_idx = np.argsort(importance)[::-1][:20]
                top_importance = importance[sorted_idx]
                top_feature_names = np.array(features)[sorted_idx]

                # Create horizontal bar plot
                ax.barh(top_feature_names, top_importance, color="royalblue")
                ax.set_title(f"{name} - Top 20 Features")
                ax.invert_yaxis()

                # Add value labels
                for j, v in enumerate(top_importance):
                    ax.text(v, j, f' {v:.3f}', va='center')
            else:
                ax.text(0.5, 0.5, f"{name} does not support feature importance",
                        ha='center', va='center', color='red')
                ax.axis('off')

        fig.tight_layout()
        plt.savefig(f'{ds_name}_feature_importance_{start//models_per_figure}split.png',
                   bbox_inches='tight', dpi=300)
        plt.close(fig)

def get_feature_importance(model, X_train, y_train):
    """
        Calculate feature importance for a given model.

        Parameters:
            model: Fitted model object
            X_train: Training features
            y_train: Training target

        Returns:
            array-like: Feature importance scores or None if not supported
        """
    model.fit(X_train, y_train)
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    elif hasattr(model, "coef_"):
        return np.abs(model.coef_).mean(axis=0)
    return None

def important_features(ds_name, models, features, X_train, X_test, y_train, y_test, top_n=20):
    """
    Analyze feature importance across models and select top features.

    This function:
    1. Evaluates each model's performance
    2. Calculates feature importance for each model
    3. Selects top features based on the best performing model
    4. Creates visualization of feature importance

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
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        model_performance[name] = accuracy

        if hasattr(model, "feature_importances_"):
            feature_importance[name] = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_)
            if importance.ndim > 1:
                importance = importance.mean(axis=0)
            feature_importance[name] = importance
        elif isinstance(model, StackingClassifier):
            stacked_importance = []
            for estimator in model.estimators_:
                if isinstance(estimator, tuple):
                    base_name, base_model = estimator
                else:
                    base_name, base_model = str(type(estimator).__name__), estimator

                if hasattr(base_model, "feature_importances_"):
                    # Handle potential shape differences
                    importance = base_model.feature_importances_
                    if importance.ndim > 1:
                        importance = importance.mean(axis=0)
                    stacked_importance.append(importance)
                elif hasattr(base_model, "coef_"):
                    importance = np.abs(base_model.coef_)
                    if importance.ndim > 1:
                        importance = importance.mean(axis=0)
                    stacked_importance.append(importance)

            if stacked_importance:
                # Convert to list of arrays with consistent shapes
                arrays = []
                for imp in stacked_importance:
                    if imp.ndim > 1:
                        arrays.append(imp.mean(axis=0))
                    else:
                        arrays.append(imp)

                # Stack the arrays
                stacked_importance = np.stack(arrays, axis=0)
                feature_importance[name] = np.mean(stacked_importance, axis=0)

    # Choose best model based on accuracy
    best_model_name = max(model_performance, key=model_performance.get)
    print(f"Best model based on accuracy: {best_model_name} with accuracy: {model_performance[best_model_name]:.2f}")

    # Get feature importance from best model
    importance = feature_importance[best_model_name]
    sorted_idx = np.argsort(importance)[::-1][:20]
    selected_features = np.array(features)[sorted_idx]

    # Filter datasets to use only top 20 features
    X_train_selected = X_train[:, sorted_idx]
    X_test_selected = X_test[:, sorted_idx]

    # Visualize the top N feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(selected_features[::-1], importance[sorted_idx][::-1], color="skyblue")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Most Important Features\n(Based on {best_model_name})")
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{ds_name}_important_feature_{best_model_name}split.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    return X_train_selected, X_test_selected, selected_features, best_model_name



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

    # Add feature importance analysis
    print("\nAnalyzing feature importance...")
    X_train_selected, X_test_selected, selected_features, best_model_name = important_features(
        ds_name, models, features, X_train, X_test, y_train, y_test, top_n=20
    )

    print(f"Best model: {best_model_name}")
    print(f"Selected top features: {selected_features}")

    print("\nTesting models with selected features...")
    test_models(ds_name, models, X_train_selected, X_test_selected, y_train, y_test, selected_features)

    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()