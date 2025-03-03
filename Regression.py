import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# # Load dataset, preprocess, and split into train/test sets
# def load_and_prepare_data():
#     ds_name = input("Enter the dataset name (without .csv): ")
#
#     try:
#         dataset = pd.read_csv(ds_name + '.csv')
#         print(f"Dataset '{ds_name}' loaded successfully.")
#         print(f"Shape: {dataset.shape}")
#
#         # User selects target column
#         print("\nAvailable columns:", dataset.columns.tolist())
#         target_column = input("Enter the target column name: ")
#
#         if target_column not in dataset.columns:
#             print(f"Error: '{target_column}' not found in dataset.")
#             return None, None, None, None
#
#         features = [col for col in dataset.columns if col != target_column]
#
#         X = dataset[features].values
#         y = dataset[target_column].values
#
#         # Standardize features
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
#
#         # Train-Test Split (80% Train, 20% Test)
#         X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=77)
#
#         return X_train, X_test, y_train, y_test, features

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
        train_data = dataset.iloc[:split_index]
        test_data = dataset.iloc[split_index:]
        print("train data: ", train_data)
        print("test data: ", test_data)

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

        return X_train_scaled, X_test_scaled, y_train, y_test, features



    except FileNotFoundError:
        print(f"Error: The file {ds_name}.csv was not found.")
        return None, None, None, None, None


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

def get_models():
    # List of all machine learning algorithms used
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=0.01, max_iter=10000),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
        "Decision Tree": DecisionTreeRegressor(random_state=77),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=77, verbose=0),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=77),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=77),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=77),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=77),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=77, verbose=0),
        "CatBoost": CatBoostRegressor(iterations=100, random_state=77, verbose=0),
        "SVR (RBF Kernel)": SVR(kernel="rbf"),
        "SVR (Linear Kernel)": SVR(kernel="linear"),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        #Current issue with this algorithm (WIP)
        #"MLP (Neural Network)": MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                                             #max_iter=2000, random_state=42, learning_rate_init=0.001, n_iter_no_change=10)
    }


    # # Define base models for stacking
    # base_models = [
    #      ("xgb", XGBRegressor(n_estimators=100, random_state=77)),
    #      ("lgbm", LGBMRegressor(n_estimators=100, random_state=77, verbose=0)),
    #      ("cat", CatBoostRegressor(iterations=100, random_state=77, verbose=0)),
    #      ("gbdt", GradientBoostingRegressor(n_estimators=100, random_state=77)),
    #      ("rf", RandomForestRegressor(n_estimators=100, random_state=77)),
    #      ("svr", SVR(kernel="rbf"))
    # ]
    #
    # # Define the meta-model (final estimator)
    # meta_model = Ridge(alpha=0.01, max_iter=10000)
    #
    # # Define the Stacking Ensemble
    # stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
    #
    # # Add Stacking model to the list of models
    # models["Stacking Ensemble"] = stacking_model

    return models

def validate_models(models, X_train, y_train, cv_splits=10):
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


def test_models(models, X_train, X_test, y_train, y_test, features):
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


            # Collect Confusion Matrix
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
        # print("model name: ", model)
        # print("model_performance", model_performance)
        # If the model has feature importances, collect them
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


# # Categorize the predicted and actual values into categories
def categorize_box_office(value):
    # if value > 1:
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

    # elif value < 1:
    #     if value < 0.2:
    #         return "Flop"
    #     elif value < 0.4:
    #         return "Below Average"
    #     elif value < 0.6:
    #         return "Average"
    #     elif value < 0.8:
    #         return "Hit"
    #     else:
    #         return "Blockbuster"
# def categorize_box_office(value):
#     percentiles = np.percentile([20, 40, 60, 80])  # Get dataset-specific thresholds
#     if value < percentiles[0]:
#         return "Flop"
#     elif value < percentiles[1]:
#         return "Below Average"
#     elif value < percentiles[2]:
#         return "Average"
#     elif value < percentiles[3]:
#         return "Hit"
#     else:
#         return "Blockbuster"

# Calculate one-away accuracy
def one_away_accuracy(y_true, y_pred):
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
    true_categories = [categorize_box_office(value) for value in y_true]
    pred_categories = [categorize_box_office(value) for value in y_pred]

    correct = sum(true == pred for true, pred in zip(true_categories, pred_categories))
    category_map = {'Flop': 0, 'Below Average': 1, 'Average': 2, 'Hit': 3, 'Blockbuster': 4}

    true_numeric = np.array([category_map[cat] for cat in true_categories])
    pred_numeric = np.array([category_map[cat] for cat in pred_categories])

    return correct / len(y_true) * 100

# # Plot category comparison
# def plot_category_comparison(ax, y_true, y_pred):
#     print(f"y_true: {y_true[:5]}, y_pred: {y_pred[:5]}")
#     true_categories = [categorize_box_office(value) for value in y_true]
#     pred_categories = [categorize_box_office(value) for value in y_pred]
#
#     category_order = ['Flop', 'Below Average', 'Average', 'Hit', 'Blockbuster']
#     true_counts = pd.Series(true_categories).value_counts().reindex(category_order, fill_value=0)
#     pred_counts = pd.Series(pred_categories).value_counts().reindex(category_order, fill_value=0)
#     print(f"Mapped true categories: {true_categories[:5]}")
#     print(f"Mapped predicted categories: {pred_categories[:5]}")
#
#     x = np.arange(len(category_order))
#
#     ax.bar(x - 0.2, true_counts, width=0.4, label='Actual', alpha=0.75)
#     ax.bar(x + 0.2, pred_counts, width=0.4, label='Predicted', alpha=0.75)
#
#     ax.set_title('Actual vs Predicted Categories')
#     ax.set_xlabel('Category')
#     ax.set_ylabel('Count')
#     ax.set_xticks(x)
#     ax.set_xticklabels(category_order)
#     ax.legend()
#
# def plot_model_results(models, plot_data, plot_category_comparison):
#     num_models = len(models)
#     models_per_figure = 6
#     model_names = list(models.keys())
#     print(f"Number of models: {len(models)}")
#     print(f"Plot data length: {len(plot_data)}")
#     print(f"Sample plot_data: {plot_data[:2]}")  # Show a sample of the data
#
#     for start in range(0, num_models, models_per_figure):
#         end = min(start + models_per_figure, num_models)
#         subset_models = model_names[start:end]
#     for start in range(0, len(plot_data), models_per_figure):
#         end = min(start + models_per_figure, len(plot_data))
#         subset_plot_data = plot_data[start:end]
#
#         # Num of columns per row
#         cols = 3
#         # Ceiling division to get row count
#         rows = -(-len(subset_plot_data) // cols)
#         # Allows for growing number of MLA
#         fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
#         axes = axes.flatten()
#
#         for i, (name, y_true, y_pred) in enumerate(subset_plot_data):
#             # Plot category comparison in the assigned subplot
#             plot_category_comparison(axes[i], y_true, y_pred)
#             axes[i].set_title(name)
#
#         # Hide unused subplots in the last figure
#         for j in range(i + 1, len(axes)):
#             fig.delaxes(axes[j])
#
#         plt.tight_layout()
#         plt.show()



# Store confusion matrices for all models
def collect_confusion_matrices(name, y_true, y_pred, confusion_matrices):
    category_order = ['Flop', 'Below Average', 'Average', 'Hit', 'Blockbuster']
    category_map = {label: idx for idx, label in enumerate(category_order)}

    y_true_encoded = [category_map[label] for label in [categorize_box_office(value) for value in y_true]]
    y_pred_encoded = [category_map[label] for label in [categorize_box_office(value) for value in y_pred]]

    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
    confusion_matrices.append((name, cm))  # Store model name and its confusion matrix

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
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=category_order)
            disp.plot(ax=axes[i], cmap="Blues", values_format="d")
            axes[i].set_title(name)

        # Hide any unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        # plt.show()


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
        # plt.show()

        return selected_features_per_model


# Main program
def main():
    X_train, X_test, y_train, y_test, features = load_and_prepare_data()

    if X_train is None or X_test is None:
        print("Exiting due to data loading error.")
        return

    models = get_models()

    print("\nValidating models with cross-validation...")
    validate_models(models, X_train, y_train)

    print("\nTesting models on the test set...")
    test_models(models, X_train, X_test, y_train, y_test, features)
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
                                             selected_features)

    # plot_model_results(models, plot_data, plot_category_comparison)


if __name__ == "__main__":
    main()
