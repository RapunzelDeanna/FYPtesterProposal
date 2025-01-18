import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import cpi
import time

start_time = time.time()
df = pd.read_csv('movies_data.csv', encoding='ISO-8859-1')

# Checks if there are missing values that could cause issues further on
missing_values = df.isnull().sum()
#print(missing_values)
# drop rows with missing values
df = df.dropna()
# check for missing values
missing_values = df.isnull().sum()
#print(missing_values)

#Adds an ID to replace the need for a name
df["id"] = df.index

df = df.iloc[:, [-1] + list(range(df.shape[1] - 1))]
#print(df.head())

# dropping irrelevant columns
#features_to_drop = ['Movie']
features_to_drop_temp = ['Movie', 'Actor 1', 'Actor 2', 'Actor 3']

# Drop the specified features
df = df.drop(features_to_drop_temp, axis=1)

label_encoder = LabelEncoder()
df['genre_encoded'] = label_encoder.fit_transform(df['Genre'])
columns_to_view = ['Genre', 'genre_encoded']  # Specify the columns you want to view
df_copy = df[columns_to_view].copy()
#print(df_copy)
# Create a mapping of each category to its encoded value
category_mapping = pd.DataFrame({
    'Genre': label_encoder.classes_,
    'Encoded': range(len(label_encoder.classes_))
})

#print("Mapping of categories to encoded values:")
#print(category_mapping)

df = df.drop('Genre', axis=1)

# Generate CPI for a range of years, e.g., from 1980 to 2020
years = list(range(1929, 2017))  # Example range of years
cpi_values = {}

# Fetch CPI for each year
for year in years:
    try:
        cpi_values[year] = cpi.get(year)
    except Exception as e:
        cpi_values[year] = None  # Handle missing data if necessary
        print(f"Error fetching CPI for {year}: {e}")

# Convert the dictionary into a DataFrame to display it as a key
cpi_key_df = pd.DataFrame(list(cpi_values.items()), columns=['Year', 'CPI'])

# Display the CPI key
print(cpi_key_df)

# Function to adjust movie budgets, earnings and box office numbers based on CPI
def adjust_for_inflation(row, base_year=2016):
    # Ensure 'Release year' is an integer
    release_year = int(row['Release year'])  # Convert to int if necessary

    # Get the CPI for the release year
    release_year_cpi = cpi.get(release_year)

    # Get the CPI for the base year (2016)
    base_year_cpi = cpi.get(base_year)

    # Adjust the budget
    adjusted_budget = row['Budget'] * (base_year_cpi / release_year_cpi)

    # Adjust the earnings
    adjusted_earnings = row['Earnings'] * (base_year_cpi / release_year_cpi)

    # Adjust the box office
    adjusted_box_office = row['Box Office'] * (base_year_cpi / release_year_cpi)

    return pd.Series([adjusted_budget, adjusted_earnings, adjusted_box_office])

# Convert 'Release year' to integer if necessary
df['Release year'] = pd.to_numeric(df['Release year'], errors='coerce', downcast='integer')

# Apply the function to adjust the budget, earnings, and box office
df[['AdjBudget', 'AdjEarnings', 'AdjBoxOffice']] = df.apply(adjust_for_inflation, axis=1)
feature_drop = ['Budget', 'Earnings', 'Box Office']

# Drop the specified features
df = df.drop(feature_drop, axis=1)
# View the updated DataFrame
print(df)
# Save the updated DataFrame to a CSV file
#df.to_csv('movies_inflation.csv', index=False)

print("Inflation done")

# Group by 'Director' and calculate the mean IMDb score
target_mean = df.groupby('Director')['IMDb score'].mean()

# Create a new column with mean-encoded values
df['Director_mean_target'] = df['Director'].map(target_mean)

# Global mean for the target
global_mean = df['IMDb score'].mean()

# Fill missing values in the encoded column
df['Director_mean_target'] = df['Director_mean_target'].fillna(global_mean)

# Smoothing with a regularization term
m = 5  # Smoothing factor
target_mean = df.groupby('Director')['IMDb score'].agg(['mean', 'count'])
target_mean['smoothed_mean'] = (target_mean['mean'] * target_mean['count'] + global_mean * m) / (target_mean['count'] + m)
df['Director_mean_target'] = df['Director'].map(target_mean['smoothed_mean'])

print(df[['Director', 'IMDb score', 'Director_mean_target']])

# Drop the specified features
df = df.drop(['Director'], axis=1)

print(df)

df.to_csv('movies_inflation+director.csv', index=False)
# End timer
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print("Time taken: ", elapsed_time)





