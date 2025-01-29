import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Regression import (
    load_dataset,
    auto_select_features_target,
    train_and_evaluate_models,
    categorize_box_office,
)


# Plot category comparison
def plot_category_comparison(ax, y_true, y_pred):
    true_categories = [categorize_box_office(value) for value in y_true]
    pred_categories = [categorize_box_office(value) for value in y_pred]

    category_order = ["Flop", "Below Average", "Average", "Hit", "Blockbuster"]
    true_counts = pd.Series(true_categories).value_counts().reindex(category_order, fill_value=0)
    pred_counts = pd.Series(pred_categories).value_counts().reindex(category_order, fill_value=0)

    x = np.arange(len(category_order))
    ax.bar(x - 0.2, true_counts, width=0.4, label="Actual", alpha=0.75)
    ax.bar(x + 0.2, pred_counts, width=0.4, label="Predicted", alpha=0.75)

    ax.set_title("Actual vs Predicted Categories")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(category_order)
    ax.legend()


# Plot feature importance for a given model
def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, "feature_importances_"):  # Tree-based models
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):  # Linear models
        importance = np.abs(model.coef_)  # Take absolute value of coefficients
    else:
        print(f"Feature importance is not available for {model_name}.")
        return

    # Sort features by importance
    sorted_indices = np.argsort(importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importance = importance[sorted_indices]

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importance, y=sorted_features, palette="viridis")
    plt.title(f"Feature Importance for {model_name}")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()


# Main visualization script
if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset()
    if dataset is None:
        exit()

    # Automatically select features and the target column
    features, target_column = auto_select_features_target(dataset)
    if not features or not target_column:
        exit()

    # Train models and get results
    results = train_and_evaluate_models(dataset, features, target_column)
    num_models = len(results)

    models_per_figure = 6  # Limit to 6 models per figure
    model_names = list(results.keys())

    for start in range(0, num_models, models_per_figure):
        end = min(start + models_per_figure, num_models)
        subset_models = model_names[start:end]

        cols = 3  # Fixed columns per figure
        rows = -(-len(subset_models) // cols)  # Ceiling division to determine rows

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axes = axes.flatten()

        for i, name in enumerate(subset_models):
            result = results[name]
            print(f"\n{name} Results:")
            for metric, value in result.items():
                if metric not in ["Predictions", "True Values", "Model"]:
                    print(f"  {metric}: {value:.2f}")

            # Plot category comparison
            plot_category_comparison(axes[i], result["True Values"], result["Predictions"])
            axes[i].set_title(name)

            # Plot feature importance
            plot_feature_importance(
                model=result["Model"],
                feature_names=features,
                model_name=name,
            )

        # Hide unused subplots if any
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
