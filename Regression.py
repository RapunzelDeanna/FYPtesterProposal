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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import StackingRegressor

# Load dataset
def load_dataset():
    ds_name = input("Enter the dataset name (without .csv): ")
    try:
        dataset = pd.read_csv(ds_name + '.csv')
        print(f"Dataset '{ds_name}' loaded successfully.")
        return dataset
    except FileNotFoundError:
        print(f"Error: The file {ds_name}.csv was not found.")
        return None


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


# Categorize the predicted and actual values into categories
def categorize_box_office(value):
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
    return correct / len(y_true) * 100


# Plot category comparison
def plot_category_comparison(ax, y_true, y_pred):
    true_categories = [categorize_box_office(value) for value in y_true]
    pred_categories = [categorize_box_office(value) for value in y_pred]

    category_order = ['Flop', 'Below Average', 'Average', 'Hit', 'Blockbuster']
    true_counts = pd.Series(true_categories).value_counts().reindex(category_order, fill_value=0)
    pred_counts = pd.Series(pred_categories).value_counts().reindex(category_order, fill_value=0)

    x = np.arange(len(category_order))

    ax.bar(x - 0.2, true_counts, width=0.4, label='Actual', alpha=0.75)
    ax.bar(x + 0.2, pred_counts, width=0.4, label='Predicted', alpha=0.75)

    ax.set_title('Actual vs Predicted Categories')
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels(category_order)
    ax.legend()

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
        #plt.show()


def plot_feature_importance(feature_importance, feature_names):
    """Plot feature importance for models that support it in a 3x2 grid (6 models per figure)."""

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

        # Hide unused subplots in the last figure
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


def train_and_evaluate_models(dataset, features, target_column):
    X = dataset[features].values
    y = dataset[target_column].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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

    kf = KFold(n_splits=10, shuffle=True, random_state=77)
    num_models = len(models)

    models_per_figure = 6
    model_names = list(models.keys())

    # Store the results for plotting
    plot_data = []
    confusion_matrices = []
    feature_importance = []

    for start in range(0, num_models, models_per_figure):
        end = min(start + models_per_figure, num_models)
        subset_models = model_names[start:end]


        # Calculates mean for each model and displays
        for name in subset_models:
            model = models[name]
            print(f"\nTraining {name} with 10-Fold Cross-Validation...")

            mse_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
            r2_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2')

            mse_scores = -mse_scores
            rmse_scores = np.sqrt(mse_scores)

            mse_mean = mse_scores.mean()
            rmse_mean = rmse_scores.mean()
            r2_mean = r2_scores.mean()

            print(f"{name}:")
            print(f"  MSE (Mean): {mse_mean:.2f}")
            print(f"  RMSE (Mean): {rmse_mean:.2f}")
            print(f"  R2 (Mean): {r2_mean:.2f}")

            y_pred = cross_val_predict(model, X_scaled, y, cv=kf)
            one_away_acc = one_away_accuracy(y, y_pred)
            exact_acc = exact_accuracy(y, y_pred)

            # After evaluating performance metrics (e.g., accuracy, MSE, R^2)
            print(f"  One-Away Accuracy: {one_away_acc:.2f}%")
            print(f"  Exact Accuracy: {exact_acc:.2f}%")

            # Collect the results for later plotting
            plot_data.append((name, y, y_pred))

            # Collect confusion matrices for later plotting
            collect_confusion_matrices(name, y, y_pred, confusion_matrices)


            try:
                model.fit(X_scaled, y)  # Fit the model

                if hasattr(model, "feature_importances_"):  # Tree-based models
                    importance = model.feature_importances_
                    feature_importance.append((name, importance))
                    print(f"Feature importance for {name}: {importance}")

                elif hasattr(model, "coef_"):  # Linear models
                    importance = np.abs(model.coef_)
                    if importance.ndim > 1:  # Handle multi-output case
                        importance = importance.mean(axis=0)
                    feature_importance.append((name, importance))
                    print(f"Feature importance (coefficients) collected for {name}")

                else:
                    print(f"{name} does not have feature importance or coefficients.")

            except Exception as e:
                print(f"Error fitting model {name}: {e}")

            # Now feature_importance should contain valid data for plotting

            # Now feature_importance will hold the relevant data

            # After all models have been evaluated, plot confusion matrices
    plot_confusion_matrices(confusion_matrices)

            # Plot feature importance if any model supports it
    if feature_importance:
        plot_feature_importance(feature_importance, features)

    for start in range(0, len(plot_data), models_per_figure):
        end = min(start + models_per_figure, len(plot_data))
        subset_plot_data = plot_data[start:end]

        # Num of columns per row
        cols = 3
        # Ceiling division to get row count
        rows = -(-len(subset_plot_data) // cols)
        # Allows for growing number of MLA
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axes = axes.flatten()

        for i, (name, y_true, y_pred) in enumerate(subset_plot_data):
            # Plot category comparison in the assigned subplot
            plot_category_comparison(axes[i], y_true, y_pred)
            axes[i].set_title(name)

        # Hide unused subplots in the last figure
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        #plt.show()
        y_preds = {}

        for model_name, model in models.items():
            y_pred = cross_val_predict(model, X_scaled, y, cv=kf)
            y_preds[model_name] = y_pred  # Store predictions for each model

        # Example: Returning predictions for a specific model (adjust as needed)
        y_test = y  # Assuming y_test is just y in cross-validation


        return y, y_pred



# Main program
def main():
    dataset = load_dataset()
    if dataset is not None:
        features, target_column = auto_select_features_target(dataset)
        if features and target_column:
            train_and_evaluate_models(dataset, features, target_column)


if __name__ == "__main__":
    main()
