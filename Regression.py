import math

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

# Load dataset
def load_dataset():
    dsName = input("Enter the dataset name (without .csv): ")
    try:
        dataset = pd.read_csv(dsName + '.csv')
        print(f"Dataset '{dsName}' loaded successfully.")
        return dataset
    except FileNotFoundError:
        print(f"Error: The file {dsName}.csv was not found.")
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


def train_and_evaluate_models(dataset, features, target_column):
    X = dataset[features].values
    y = dataset[target_column].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        "CatBoost": CatBoostRegressor(iterations=100, random_state=42, verbose=0),
        "SVR (RBF Kernel)": SVR(kernel="rbf"),
        "SVR (Linear Kernel)": SVR(kernel="linear"),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "MLP (Neural Network)": MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                                             max_iter=1000, random_state=42)
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    num_models = len(models)

    models_per_figure = 6  # Set the max number of models per figure
    model_names = list(models.keys())

    for start in range(0, num_models, models_per_figure):
        end = min(start + models_per_figure, num_models)
        subset_models = model_names[start:end]

        cols = 3  # 3 columns per figure
        rows = -(-len(subset_models) // cols)  # Ceiling division to get row count

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axes = axes.flatten()

        for i, name in enumerate(subset_models):
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

            print(f"  One-Away Accuracy: {one_away_acc:.2f}%")
            print(f"  Exact Accuracy: {exact_acc:.2f}%")

            # Plot category comparison in the assigned subplot
            plot_category_comparison(axes[i], y, y_pred)
            axes[i].set_title(name)

        # Hide unused subplots in the last figure
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()



# Main program
def main():
    dataset = load_dataset()
    if dataset is not None:
        features, target_column = auto_select_features_target(dataset)
        if features and target_column:
            train_and_evaluate_models(dataset, features, target_column)


if __name__ == "__main__":
    main()
