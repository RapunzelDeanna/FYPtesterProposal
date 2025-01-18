import matplotlib.pyplot as plt
import seaborn as sns

def plot_regression_results(y_test, y_pred):
    # 1. Predicted vs Actual Values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
    plt.title('Predicted vs Actual Values')
    plt.xlabel('Actual Values (AdjBoxOffice)')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.show()

    # 2. Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.residplot(x=y_pred, y=residuals, lowess=True, color='green')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (Errors)')
    plt.grid(True)
    plt.show()

    # 3. Distribution of Errors (Residuals)
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='purple', bins=30)
    plt.title('Distribution of Errors (Residuals)')
    plt.xlabel('Residuals (Errors)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # 4. Line Plot of Actual vs Predicted Values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Values', color='red', alpha=0.7)
    plt.plot(y_pred, label='Predicted Values', color='blue', alpha=0.7)

    plt.title('Actual vs Predicted Values')
    plt.xlabel('Index')
    plt.ylabel('AdjBoxOffice')
    plt.legend()
    plt.grid(True)
    plt.show()
