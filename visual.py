import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import RegularGridInterpolator

import Regression

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
            plt.show()

            # Plot Residuals
            fig, axes = plt.subplots(2, 3, figsize=(15, 15))
            axes = axes.flatten()
            for idx, model_name in enumerate(subset_models):
                y_pred = y_preds[model_name]
                plot_residuals(y_test, y_pred, model_name, axes[idx])
            plt.tight_layout()
            plt.show()

            # Plot Distribution of Errors (Residuals)
            fig, axes = plt.subplots(2, 3, figsize=(15, 15))
            axes = axes.flatten()
            for idx, model_name in enumerate(subset_models):
                y_pred = y_preds[model_name]
                residuals = y_test - y_pred
                plot_distribution_of_errors(residuals, model_name, axes[idx])
            plt.tight_layout()
            plt.show()

            # Plot Line Plot of Actual vs Predicted Values
            fig, axes = plt.subplots(2, 3, figsize=(15, 15))
            axes = axes.flatten()
            for idx, model_name in enumerate(subset_models):
                y_pred = y_preds[model_name]
                plot_line_actual_vs_predicted(y_test, y_pred, model_name, axes[idx])
            plt.tight_layout()
            plt.show()
