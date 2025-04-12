import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import cpi
import time

# Takes a while with cpi so just checking
start_time = time.time()
df = pd.read_csv('movies_data.csv', encoding='ISO-8859-1')

# Checks if there are missing values that could cause issues later on
missing_values = df.isnull().sum()
# drop rows with missing values
df = df.dropna()
# check for missing values again
missing_values = df.isnull().sum()

#Adds an ID to replace the need for a name
df["id"] = df.index
#reorders in so ID is first in dataset
df = df.iloc[:, [-1] + list(range(df.shape[1] - 1))]

# dropping irrelevant columns
features_to_drop = ['Movie', 'Earnings', 'id']
df = df.drop(features_to_drop, axis=1)

# Converts the genres into their own feature in boolean form
label_encoder = LabelEncoder()
df['genre_encoded'] = label_encoder.fit_transform(df['Genre'])
columns_to_view = ['Genre', 'genre_encoded']
df_copy = df[columns_to_view].copy()
#print(df_copy)

# Create a mapping of each category to its encoded value
category_mapping = pd.DataFrame({
    'Genre': label_encoder.classes_,
    'Encoded': range(len(label_encoder.classes_))
})
# Drop original feature
df = df.drop('Genre', axis=1)

# Generate CPI for a range of years
years = list(range(1929, 2017))
cpi_values = {}

# Fetch CPI for each year
for year in years:
    try:
        cpi_values[year] = cpi.get(year)
    except Exception as e:
        # Handle missing data if necessary
        cpi_values[year] = None
        print(f"Error fetching CPI for {year}: {e}")

# Convert the dictionary into a DataFrame to display it as a key
cpi_key_df = pd.DataFrame(list(cpi_values.items()), columns=['Year', 'CPI'])

# keep consistency across datasets
df.rename(columns={'Release year': 'release_year'}, inplace=True)

# Display the CPI key
#print(cpi_key_df)

# Function to adjust movie budgets and box office numbers based on CPI
def adjust_for_inflation(row, base_year=2016):
    # Ensure 'Release year' is an integer
    release_year = int(row['release_year'])

    # Get the CPI for the release year
    release_year_cpi = cpi.get(release_year)

    # Get the CPI for the base year (2016)
    base_year_cpi = cpi.get(base_year)

    # Adjust the budget
    adjusted_budget = row['Budget'] * (base_year_cpi / release_year_cpi)

    # Adjust the box office
    adjusted_box_office = row['Box Office'] * (base_year_cpi / release_year_cpi)

    return pd.Series([adjusted_budget, adjusted_box_office])

# Convert 'Release year' to integer if necessary
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce', downcast='integer')

# Apply the function to adjust the budget and box office
df[['AdjBudget', 'AdjBoxOffice']] = df.apply(adjust_for_inflation, axis=1)
feature_drop = ['Budget', 'Box Office']

# Drop the specified features
df = df.drop(feature_drop, axis=1)
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









# Set the actor columns to a variable to repeat the process
actor_columns = ['Actor 1', 'Actor 2', 'Actor 3']

# Iterate over each actor column
for actor_col in actor_columns:
    # Group by the actor and calculate the mean IMDb score
    target_mean = df.groupby(actor_col)['IMDb score'].mean()

    # Create a new column with the mean-encoded values for the actor
    df[f'{actor_col}_mean_target'] = df[actor_col].map(target_mean)

    # Global mean for the target
    global_mean = df['IMDb score'].mean()

    # Fill missing values in the encoded column with the global mean
    df[f'{actor_col}_mean_target'] = df[f'{actor_col}_mean_target'].fillna(global_mean)

    # Smoothing with a regularization term
    m = 5  # Smoothing factor
    target_mean = df.groupby(actor_col)['IMDb score'].agg(['mean', 'count'])
    target_mean['smoothed_mean'] = (target_mean['mean'] * target_mean['count'] + global_mean * m) / (target_mean['count'] + m)
    df[f'{actor_col}_mean_target'] = df[actor_col].map(target_mean['smoothed_mean'])

# Display the results
#print(df[['Director', 'IMDb score', 'Actor 1', 'Actor 2', 'Actor 3', 'Actor 1_mean_target', 'Actor 2_mean_target', 'Actor 3_mean_target']])




# # Copy dataset to avoid modifying the original
# scaled_dataset = df.copy()
# # Define columns that require Min-Max Scaling
# columns_to_scale = ['Running time', 'Actors Box Office %', 'Director Box Office %',
#     'Oscar and Golden Globes nominations', 'Oscar and Golden Globes awards',
#     'IMDb score', 'AdjBudget', 'Director_mean_target',
#     'Actor 1_mean_target', 'Actor 2_mean_target', 'Actor 3_mean_target'
# ]
# # Initialize the Min-Max Scaler
# scaler = MinMaxScaler()
# # Apply scaling to the selected columns
# scaled_dataset[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
# # Verify the scaling
# print(scaled_dataset[columns_to_scale].describe())
#
# # Merge scaled values back into the original dataset
# df[columns_to_scale] = scaled_dataset[columns_to_scale]


df = df.rename(columns={'AdjBoxOffice': 'revenue'})
print(df['revenue'].describe())
print(df['revenue'].quantile([0.2, 0.4, 0.6, 0.8, 1.0]))

def categorize_box_office(value):
    if value < 5e6: # <$5M
        return 0  # Flop
    elif value < 30e6:# $5M - $30M
        return 1  # Below Average
    elif value < 100e6: # $30M - $100M
        return 2  # Average
    elif value < 300e6:# $100M - $300M
        return 3  # Hit
    else: # >$300M
        return 4  # Blockbuster


df['revenue_class'] = df['revenue'].apply(categorize_box_office)


# Drop the old features
df = df.drop(['Director', 'Actor 1', 'Actor 2', 'Actor 3', 'revenue'], axis=1)


print(df)

#Preprocessed data is saved to the prep_movies.csv file
df.to_csv('dataset1class.csv', index=False)
# End timer
end_time = time.time()

# Calculate time taken
elapsed_time = end_time - start_time
print("Time taken: ", elapsed_time)





