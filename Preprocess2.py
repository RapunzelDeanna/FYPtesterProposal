import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import time
from sklearn.preprocessing import MinMaxScaler


start_time = time.time()
df = pd.read_csv('TMBD Movie Dataset 2.csv', encoding='ISO-8859-1')

# Sort movies by id (ascending order)
df = df.sort_values(by='id', ascending=True).reset_index(drop=True)

# Checks if there are missing values that could cause issues further on
missing_values = df.isnull().sum()
#print(missing_values)
# drop rows with missing values
df = df.dropna()
# check for missing values
missing_values = df.isnull().sum()
#print(missing_values)

# dropping irrelevant columns
features_to_drop = ['Unnamed: 0', 'revenue', 'budget', 'imdb_id', 'original_title', 'homepage', 'tagline', 'overview', 'production_companies', 'profit']
df = df.drop(features_to_drop, axis=1)


# Handle multiple genres
def preprocess_genres(dataframe, genre_column):
    # Split genres into lists
    dataframe['genres'] = dataframe[genre_column].str.split('|')

    # Use MultiLabelBinarizer to one-hot encode genres
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(dataframe['genres'])
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=dataframe.index)

    # Drop the original genre column and add on the binary genre features
    dataframe = pd.concat([dataframe.drop(columns=[genre_column, 'genres']), genre_df], axis=1)
    return dataframe, mlb.classes_

# Preprocess dataset to handle multiple genres
df, unique_genres = preprocess_genres(df, 'genres')

# Convert the popularity level into int
unique_levels = df['popularity_level'].unique()
# Define the order of categories
popularity_mapping = {'High': 3, 'Moderately High': 2, 'Medium': 1, 'Low': 0}
# Map the levels to numerical values
df['popularity_encoded'] = df['popularity_level'].map(popularity_mapping)




# Convert release_date to datetime format
df['release_date'] = pd.to_datetime(df['release_date'])
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


# Calculate Star Power based on the combined revenue of each of the cast for that movie
# Split the cast column by '|' and expand into a new row for each cast member
expanded_df = df.assign(cast=df['cast'].str.split('|')).explode('cast')
# Aggregate the total revenue by cast member
star_power = expanded_df.groupby('cast')['revenue_adj'].sum().reset_index()
# Rename columns for clarity
star_power.rename(columns={'cast': 'actor', 'revenue_adj': 'star_power'}, inplace=True)
# Sort by star power in descending order
star_power = star_power.sort_values(by='star_power', ascending=False)
# Calculate total star power for each movie
# Merge star power with the original movie DataFrame
expanded_df = expanded_df.merge(star_power, left_on='cast', right_on='actor', how='left')
# Group by movie and sum the star power for each movie
movie_star_power = expanded_df.groupby('id')['star_power'].sum().reset_index()
# Merge movie star power back to the original DataFrame
df = df.merge(movie_star_power, on='id', how='left')


# Calculate Director Power based on the total revenue of all movies they directed
# Split the 'director' column by '|' and expand into multiple rows
expanded_df = df.assign(director=df['director'].str.split('|')).explode('director')

# Aggregate total revenue by each director
director_power = expanded_df.groupby('director')['revenue_adj'].sum().reset_index()

# Rename for clarity
director_power.rename(columns={'revenue_adj': 'director_power'}, inplace=True)

# Sort directors by their total revenue (Director Power)
director_power = director_power.sort_values(by='director_power', ascending=False)

# Merge Director Power back into the expanded DataFrame
expanded_df = expanded_df.merge(director_power, on='director', how='left')

# Group by movie and sum the Director Power for each movie
movie_director_power = expanded_df.groupby('id')['director_power'].sum().reset_index()

# Merge total Director Power back into the original DataFrame
df = df.merge(movie_director_power, on='id', how='left')


# Function to process multi-word phrases (replaces spaces with underscores)
def process_keywords(keywords):
    # Split the keywords by commas
    keyword_list = keywords.split(" | ")

    # Replace spaces in multi-word phrases with "_"
    keyword_list = [kw.replace(" ", "_") for kw in keyword_list]

    return " ".join(keyword_list)


# Apply the function to the keywords column
df["keywords"] = df["keywords"].apply(process_keywords)

# Apply TF-IDF transformation
tfidf = TfidfVectorizer(max_features=500, stop_words="english",
                        token_pattern=r"\b\w+\b")  # token_pattern ensures that words are captured properly
tfidf_matrix = tfidf.fit_transform(df["keywords"])

# Convert TF-IDF matrix to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

# Concatenate TF-IDF features with the original dataset
df = pd.concat([df.drop(columns=["keywords"]), tfidf_df], axis=1)  # Drop original keywords column

# Show completion message and new shape
print("TF-IDF preprocessing completed. New shape:", df.shape)
print(df.head())


# Copy dataset to avoid modifying the original
scaled_dataset = df.copy()
# Define columns that require Min-Max Scaling
columns_to_scale = [
    'popularity', 'runtime', 'vote_count',
    'budget_adj', 'popularity_encoded', 'star_power', 'director_power'
]
# Initialize the Min-Max Scaler
scaler = MinMaxScaler()
# Apply scaling to the selected columns
scaled_dataset[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
# Verify the scaling
print(scaled_dataset[columns_to_scale].describe())
# Merge scaled values back into the original dataset
df[columns_to_scale] = scaled_dataset[columns_to_scale]


# End timer
end_time = time.time()

# Calculate time taken
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
#plt.show()

df = df.drop(['popularity_level', 'release_date', 'release_day_name', 'cast', 'director'], axis=1)

#Preprocessed data is saved to the df.csv file (for faster testing purposes)
df.to_csv('df.csv', index=False)






