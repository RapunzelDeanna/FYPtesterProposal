import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import Regression
import pandas as pd


def plot_category_comparison(ax, y_true, y_pred):
    print(f"y_true: {y_true[:5]}, y_pred: {y_pred[:5]}")

    true_categories = np.array([Regression.categorize_box_office(value) for value in y_true])
    pred_categories = np.array([Regression.categorize_box_office(value) for value in y_pred])

    category_order = ['Flop', 'Below Average', 'Average', 'Hit', 'Blockbuster']

    # Count actual movies in each category
    true_counts = pd.Series(true_categories).value_counts().reindex(category_order, fill_value=0)

    # Count total movies predicted in each category (regardless of correctness)
    predicted_counts = pd.Series(pred_categories).value_counts().reindex(category_order, fill_value=0)

    # Count correctly classified movies per predicted category
    correct_counts = pd.Series(
        [pred for true, pred in zip(true_categories, pred_categories) if true == pred]
    ).value_counts().reindex(category_order, fill_value=0)

    # Misclassified counts = Total predicted in category - Correctly classified
    misclassified_counts = predicted_counts - correct_counts

    x = np.arange(len(category_order))
    width = 0.4  # Bar width for side-by-side comparison

    # **Actual category distribution (True counts)**
    ax.bar(x - width / 2, true_counts, width=width, color='blue', alpha=0.6, label='Actual Count')

    # **Total Predicted Category Distribution (Correct + Misclassified)**
    ax.bar(x + width / 2, correct_counts, width=width, color='green', label='Correctly Classified')
    ax.bar(x + width / 2, misclassified_counts, width=width, color='red', label='Misclassified', bottom=correct_counts)

    ax.set_title('Actual vs Predicted Categories')
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels(category_order)
    ax.legend()


# Plot results for multiple models
def plot_model_results(models, plot_data):
    num_models = len(models)
    models_per_figure = 6

    for start in range(0, len(plot_data), models_per_figure):
        end = min(start + models_per_figure, len(plot_data))
        subset_plot_data = plot_data[start:end]

        cols = 3
        rows = -(-len(subset_plot_data) // cols)  # Ceiling division
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axes = axes.flatten()

        for i, (name, y_true, y_pred) in enumerate(subset_plot_data):
            plot_category_comparison(axes[i], y_true, y_pred)  # Make sure data is passed correctly
            axes[i].set_title(name)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])  # Remove unused axes

        plt.tight_layout()
        plt.show()


def plot_predicted_vs_actual(y_test, y_pred, model_name, ax):
    """Plots Predicted vs Actual values for a given model on the provided axes."""
    ax.scatter(y_test, y_pred, color='blue', alpha=0.6)
    ax.set_title(f'Predicted vs Actual Values - {model_name}')
    ax.set_xlabel('Actual Values (AdjBoxOffice)')
    ax.set_ylabel('Predicted Values')
    ax.grid(True)

def plot_residuals(y_test, y_pred, model_name, ax):
    """Plots Residuals for a given model on the provided axes."""
    residuals = y_test - y_pred
    sns.residplot(x=y_pred, y=residuals, lowess=True, color='green', ax=ax)
    ax.set_title(f'Residual Plot - {model_name}')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals (Errors)')
    ax.grid(True)

def plot_distribution_of_errors(residuals, model_name, ax):
    """Plots Distribution of Errors for a given model on the provided axes."""
    sns.histplot(residuals, kde=True, color='purple', bins=30, ax=ax)
    ax.set_title(f'Distribution of Errors - {model_name}')
    ax.set_xlabel('Residuals (Errors)')
    ax.set_ylabel('Frequency')
    ax.grid(True)

def plot_line_actual_vs_predicted(y_test, y_pred, model_name, ax):
    """Plots Line plot of Actual vs Predicted values for a given model on the provided axes."""
    ax.plot(y_test, label='Actual Values', color='red', alpha=0.7)
    ax.plot(y_pred, label='Predicted Values', color='blue', alpha=0.7)
    ax.set_title(f'Actual vs Predicted Values - {model_name}')
    ax.set_xlabel('Index')
    ax.set_ylabel('AdjBoxOffice')
    ax.legend()
    ax.grid(True)

# Load dataset
X_train, X_test, y_train, y_test, features = Regression.load_and_prepare_data()
models = Regression.get_models()

if X_train is not None and X_test is not None and y_train is not None and y_test is not None and models is not None:
        y_test, y_preds, plot_data = Regression.test_models(models, X_train, X_test, y_train, y_test, features)

        # Create grids for each type of plot
        models = list(y_preds.keys())
        models_per_figure = 6
        num_models = len(models)

        for start in range(0, num_models, models_per_figure):
            end = min(start + models_per_figure, num_models)
            subset_models = models[start:end]

            # Plot Predicted vs Actual Values
            fig, axes = plt.subplots(2, 3, figsize=(15, 15))
            axes = axes.flatten()
            for idx, model_name in enumerate(subset_models):
                y_pred = y_preds[model_name]
                plot_predicted_vs_actual(y_test, y_pred, model_name, axes[idx])
            plt.tight_layout()
            # plt.show()

            # Plot Residuals
            fig, axes = plt.subplots(2, 3, figsize=(15, 15))
            axes = axes.flatten()
            for idx, model_name in enumerate(subset_models):
                y_pred = y_preds[model_name]
                plot_residuals(y_test, y_pred, model_name, axes[idx])
            plt.tight_layout()
            # plt.show()

            # Plot Distribution of Errors (Residuals)
            fig, axes = plt.subplots(2, 3, figsize=(15, 15))
            axes = axes.flatten()
            for idx, model_name in enumerate(subset_models):
                y_pred = y_preds[model_name]
                residuals = y_test - y_pred
                plot_distribution_of_errors(residuals, model_name, axes[idx])
            plt.tight_layout()
            # plt.show()

            # Plot Line Plot of Actual vs Predicted Values
            fig, axes = plt.subplots(2, 3, figsize=(15, 15))
            axes = axes.flatten()
            for idx, model_name in enumerate(subset_models):
                y_pred = y_preds[model_name]
                plot_line_actual_vs_predicted(y_test, y_pred, model_name, axes[idx])
            plt.tight_layout()
            # plt.show()

            

        plot_model_results(models, plot_data)
