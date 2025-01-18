import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from math import sqrt
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
    # Get the target column from the user
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

dataset = load_dataset()
if dataset is not None:
    features, target_column = auto_select_features_target(dataset)

    if features and target_column:
        X = dataset[features].values
        y = dataset[target_column].values

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "CatBoost": CatBoostRegressor(iterations=100, learning_rate=0.1, random_state=42, verbose=0),
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1)
}


# Define categories for box office performance
def categorize_box_office(values):
    categories = []
    for value in values:
        if value < 1e6:
            categories.append("Flop")
        elif value < 5e6:
            categories.append("Below Average")
        elif value < 40e6:
            categories.append("Average")
        elif value < 150e6:
            categories.append("Hit")
        else:
            categories.append("Blockbuster")
    return np.array(categories)


# Categorize the actual test values
actual_categories = categorize_box_office(y_test)

# Train and evaluate each model
results = {}
category_distributions = {}
accuracy_results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Predict

    # Evaluate regression performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Categorize predictions
    predicted_categories = categorize_box_office(y_pred)

    # Calculate category distribution
    actual_distribution = pd.Series(actual_categories).value_counts()
    predicted_distribution = pd.Series(predicted_categories).value_counts()

    # Calculate accuracy of categorization
    accuracy = np.mean(predicted_categories == actual_categories)

    # Store results
    results[name] = {"MSE": mse, "RMSE": rmse, "R2": r2}
    category_distributions[name] = (actual_distribution, predicted_distribution)
    accuracy_results[name] = accuracy

    print(f"Results for {name}:")
    print(f"  MSE: {mse}")
    print(f"  RMSE: {rmse}")
    print(f"  R2: {r2}")
    print(f"  Accuracy: {accuracy:.2%}")

# Print all results for comparison
print("\nSummary of Results:")
for name, metrics in results.items():
    print(
        f"{name}: MSE = {metrics['MSE']}, RMSE = {metrics['RMSE']}, R2 = {metrics['R2']}, Accuracy = {accuracy_results[name]:.2%}")

# Define category order
category_order = ["Flop", "Below Average", "Average", "Hit", "Blockbuster"]
category_to_index = {category: idx for idx, category in enumerate(category_order)}

# Initialize dictionaries to store results
one_away_accuracies = {}

for name, model in models.items():
    # Predict categories for this model
    y_pred = model.predict(X_test)
    predicted_categories = categorize_box_office(y_pred)
    actual_categories = categorize_box_office(y_test)

    # Convert categories to indices
    predicted_indices = [category_to_index[cat] for cat in predicted_categories]
    actual_indices = [category_to_index[cat] for cat in actual_categories]

    # Calculate exact matches
    exact_matches = sum(1 for actual, pred in zip(actual_indices, predicted_indices) if actual == pred)
    exact_accuracy = exact_matches / len(y_test) * 100

    # Calculate one-away matches
    one_away_matches = sum(1 for actual, pred in zip(actual_indices, predicted_indices)
                           if abs(actual - pred) <= 1)
    one_away_accuracy = one_away_matches / len(y_test) * 100

    # Store accuracies
    one_away_accuracies[name] = {"Exact Accuracy": exact_accuracy, "One-Away Accuracy": one_away_accuracy}

    # Print model results
    print(f"\n{name}:")
    print(f"  Exact Accuracy: {exact_accuracy:.2f}%")
    print(f"  One-Away Accuracy: {one_away_accuracy:.2f}%")

    # Plot bar chart for actual vs predicted categories
    predicted_distribution = pd.Series(predicted_categories).value_counts().reindex(category_order, fill_value=0)
    actual_distribution = pd.Series(actual_categories).value_counts().reindex(category_order, fill_value=0)
    distribution_df = pd.DataFrame({'Actual': actual_distribution, 'Predicted': predicted_distribution})

    plt.figure(figsize=(12, 6))
    distribution_df.plot(kind='bar', alpha=0.75, width=0.8, ax=plt.gca())
    plt.title(f"Actual vs. Predicted Box Office Categories for {name}")
    plt.ylabel("Count")
    plt.xlabel("Category")
    plt.xticks(rotation=45)
    plt.legend(["Actual", "Predicted"])
    plt.tight_layout()
    plt.show()

# Print overall summary of one-away accuracies
print("\nOne-Away Accuracy Summary:")
for name, accuracies in one_away_accuracies.items():
    print(f"{name}: Exact Accuracy = {accuracies['Exact Accuracy']:.2f}%, "
          f"One-Away Accuracy = {accuracies['One-Away Accuracy']:.2f}%")
