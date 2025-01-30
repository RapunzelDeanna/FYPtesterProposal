import cpi
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MultiLabelBinarizer
import time
# Takes a while to run so this just checks
start_time = time.time()
df = pd.read_csv('TMDB_movie_dataset_v11.csv', encoding='ISO-8859-1')

#removes all adult movies
df = df[~df['adult']]

# dropping irrelevant columns
features_to_drop = ['status', 'backdrop_path', 'homepage', 'imdb_id', 'original_title', 'overview', 'poster_path', 'tagline']
df = df.drop(features_to_drop, axis=1)


# Checks if there are missing values that could cause issues later on
missing_values = df.isnull().sum()
#print(missing_values)
# drop rows with missing values
df = df.dropna()
# check for missing values again to confirm
missing_values = df.isnull().sum()

# Handle multiple genres
def preprocess_genres(dataframe, genre_column):
    # Split genres into lists
    dataframe['genres'] = dataframe[genre_column].str.split(', ')

    # Use MultiLabelBinarizer to one-hot encode genres
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(dataframe['genres'])
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=dataframe.index)

    # Drop the original genre column and add on the binary genre features
    dataframe = pd.concat([dataframe.drop(columns=[genre_column, 'genres']), genre_df], axis=1)
    return dataframe, mlb.classes_


# Preprocess dataset to handle multiple genres
df, unique_genres = preprocess_genres(df, 'genres')
#print(unique_genres)


# Convert release_date to datetime format
df['release_date'] = pd.to_datetime(df['release_date'])
# Extract release year
df['release_year'] = df['release_date'].dt.year
# Extract release month and release day
df['release_month'] = df['release_date'].dt.month
# Extract the day of the week (0 = Monday, 6 = Sunday)
df['release_day_of_week'] = df['release_date'].dt.dayofweek

# Extract the day of the week as the full name
df['release_day_name'] = df['release_date'].dt.day_name()
# Group by day of the week and count the number of movies
day_counts = df['release_day_name'].value_counts()

# Sort days of the week in order
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_counts = day_counts.reindex(day_order)
df = df.drop(['release_day_name'], axis=1)




#Work in progress from here







min_year = df['release_year'].min()
max_year = df['release_year'].max()

# Get movies from the earliest year
earliest_movies = df[df['release_year'] == min_year]
# Get movies from the latest year
latest_movies = df[df['release_year'] == max_year]

print(f"Movies from the earliest year ({min_year}):")
print(earliest_movies)

print(f"\nMovies from the latest year ({max_year}):")
print(latest_movies)

# Function to adjust movie budgets, earnings and box office numbers based on CPI
def adjust_for_inflation(row, base_year=2020):
    # Ensure 'Release year' is an integer
    release_year = int(row['release_year'])  # Convert to int if necessary

    # Get the CPI for the release year
    release_year_cpi = cpi.get(release_year)

    # Get the CPI for the base year (2024)
    base_year_cpi = cpi.get(base_year)

    # Adjust the budget
    adjusted_budget = row['budget'] * (base_year_cpi / release_year_cpi)

    # Adjust the revenue
    adjusted_revenue = row['revenue'] * (base_year_cpi / release_year_cpi)

    return pd.Series([adjusted_budget, adjusted_revenue])

# Convert 'Release year' to integer if necessary
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce', downcast='integer')

# Apply the function to adjust the budget, earnings, and box office
df[['AdjBudget', 'Adjrevenue']] = df.apply(adjust_for_inflation, axis=1)
feature_drop = ['budget', 'revenue']
print(df)





# Preprocessed file is saved as Movies1M.csv
df.to_csv('Movies1M.csv', index=False)

# End timer
end_time = time.time()

# Calculate time taken to run
elapsed_time = end_time - start_time
print("Time taken: ", elapsed_time)

# Plot the day of the week graph
plt.figure(figsize=(10, 6))
day_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Number of Movies Released by Day of the Week', fontsize=16)
plt.xlabel('Day of the Week', fontsize=14)
plt.ylabel('Number of Movies', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()






