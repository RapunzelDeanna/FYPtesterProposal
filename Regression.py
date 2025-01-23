import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler

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
        return"Below Average"
    elif value < 40e6:
        return"Average"
    elif value < 150e6:
        return"Hit"
    else:
        return"Blockbuster"

def one_away_accuracy(y_true, y_pred):
    """Calculate the one-away accuracy: checks if the prediction is within one category away from the true value."""
    correct = 0
    total = len(y_true)

    # Convert both true values and predictions into categories
    true_categories = [categorize_box_office(value) for value in y_true]
    pred_categories = [categorize_box_office(value) for value in y_pred]

    # Define a mapping of categories to numeric values for comparison
    category_map = {'Flop': 0, 'Below Average': 1, 'Average': 2, 'Hit': 3, 'Blockbuster': 4}

    for true, pred in zip(true_categories, pred_categories):
        if abs(category_map[true] - category_map[pred]) <= 1:
            correct += 1

    return correct / total * 100  # Return percentage of one-away matches

def exact_accuracy(y_true, y_pred):
    """Calculate the exact accuracy: checks if the predicted category exactly matches the true value."""
    correct = 0
    total = len(y_true)

    # Convert both true values and predictions into categories
    true_categories = [categorize_box_office(value) for value in y_true]
    pred_categories = [categorize_box_office(value) for value in y_pred]

    # Check if the categories exactly match
    for true, pred in zip(true_categories, pred_categories):
        if true == pred:
            correct += 1

    return correct / total * 100  # Return percentage of exact matches

def plot_category_comparison(ax, y_true, y_pred):
    # Convert y_true and y_pred to categories
    true_categories = [categorize_box_office(value) for value in y_true]
    pred_categories = [categorize_box_office(value) for value in y_pred]

    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame({
        'Actual': true_categories,
        'Predicted': pred_categories
    })

    # Ensure category_order is in the correct order
    category_order = ['Flop', 'Below Average', 'Average', 'Hit', 'Blockbuster']

    # Count the occurrences of each category in the actual and predicted values
    actual_counts = comparison_df['Actual'].value_counts().reindex(category_order, fill_value=0)
    predicted_counts = comparison_df['Predicted'].value_counts().reindex(category_order, fill_value=0)

    # Set the x positions for each category
    x = np.arange(len(category_order))

    # Plot bars for actual categories
    ax.bar(x - 0.2, actual_counts, width=0.4, label='Actual', alpha=0.75)

    # Plot bars for predicted categories, shifted to the right
    ax.bar(x + 0.2, predicted_counts, width=0.4, label='Predicted', alpha=0.75)

    # Adding labels and title
    ax.set_title('Actual vs Predicted Categories')
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels(category_order)

    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45)

    # Display the legend
    ax.legend()

# Main program
dataset = load_dataset()
if dataset is not None:
    features, target_column = auto_select_features_target(dataset)

    if features and target_column:
        X = dataset[features].values
        y = dataset[target_column].values

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1),
    "CatBoost": CatBoostRegressor(iterations=100, learning_rate=0.1, random_state=42, verbose=0),
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1)
}

# Define KFold for 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize dictionary to store cross-validation results
cv_results = {}

# Create a grid layout for the plots (2 rows and 3 columns for 6 models)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()  # Flatten the 2D array to make indexing easier

fig.tight_layout(pad=5.0)

# Perform 10-fold cross-validation for each model and plot
for i, (name, model) in enumerate(models.items()):
    print(f"\nTraining {name} with 10-Fold Cross-Validation...")

    # Cross-validation MSE (negative due to scoring format) and accuracy calculation
    mse_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2')

    # Convert MSE to positive
    mse_scores = -mse_scores

    # Calculate RMSE from MSE
    rmse_scores = np.sqrt(mse_scores)

    # Calculate mean values
    mse_mean = mse_scores.mean()
    rmse_mean = rmse_scores.mean()
    r2_mean = r2_scores.mean()

    # Store cross-validation results
    cv_results[name] = {
        "MSE (Mean)": mse_mean,
        "RMSE (Mean)": rmse_mean,
        "R2 (Mean)": r2_mean
    }

    print(f"{name}:")
    print(f"  MSE (Mean): {mse_mean:.2f}")
    print(f"  RMSE (Mean): {rmse_mean:.2f}")
    print(f"  R2 (Mean): {r2_mean:.2f}")

    # One-away accuracy calculation after cross-validation (using the model on each fold)
    y_pred = cross_val_predict(model, X_scaled, y, cv=kf)
    one_away_acc = one_away_accuracy(y, y_pred)
    print(f"  One-Away Accuracy: {one_away_acc:.2f}%")

    # Exact accuracy calculation after cross-validation (using the model on each fold)
    exact_acc = exact_accuracy(y, y_pred)
    print(f"  Exact Accuracy: {exact_acc:.2f}%")

    # Plot category comparison for actual vs predicted values in the respective subplot
    plot_category_comparison(axes[i], y, y_pred)

plt.show()
