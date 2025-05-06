import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Regression

def plot_category_comparison(ax, y_true, y_pred):
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
def plot_model_results(models, plot_data, ds_name):
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
            plot_category_comparison(axes[i], y_true, y_pred)
            axes[i].set_title(name)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(f'{ds_name}model_comparison_{start//models_per_figure}.png',
                   bbox_inches='tight', dpi=300)
        plt.close()

# Load dataset
X_train, X_test, y_train, y_test, features, ds_name = Regression.load_and_prepare_data()
models = Regression.get_models(X_train, y_train)

if X_train is not None and X_test is not None and y_train is not None and y_test is not None and models is not None:
        y_test, y_preds, plot_data = Regression.test_models(ds_name, models, X_train, X_test, y_train, y_test, features)


        # Create grids for each type of plot
        models = list(y_preds.keys())
        models_per_figure = 6
        num_models = len(models)

        for start in range(0, num_models, models_per_figure):
            end = min(start + models_per_figure, num_models)
            subset_models = models[start:end]
            

        plot_model_results(models, plot_data, ds_name)
