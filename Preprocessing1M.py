import cpi
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MultiLabelBinarizer
import time
from sklearn.feature_extraction.text import TfidfVectorizer

# Takes a while to run so this just checks
start_time = time.time()
df = pd.read_csv('Movies1M.csv', encoding='ISO-8859-1')
df = df.head(2000)
#removes all adult movies
df = df[~df['adult']]

status_counts_before = df['status'].value_counts()
print("Status counts before filtering:\n", status_counts_before, "\n")

df = df[df['status'] == 'Released']
# id,title,vote_average,vote_count,release_date,revenue,runtime,adult,budget,original_language,popularity,production_companies,
# production_countries,spoken_languages,keywords,Action,Adventure,Animation,Comedy,Crime,Documentary,Drama,Family,Fantasy,
# History,Horror,Music,Mystery,Romance,Science Fiction,TV Movie,Thriller,War,Western,release_year,release_month,release_day_of_week,AdjBudget,Adjrevenue
# dropping irrelevant columns
features_to_drop = ['status', 'backdrop_path', 'homepage', 'imdb_id', 'original_title', 'overview', 'poster_path', 'tagline', 'adult', 'title', 'production_countries', 'spoken_languages']
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







df = df[(df['release_year'] >= 1913) & (df['release_year'] <= 2020)]
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
def adjust_for_inflation(row, base_year=2010):
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
df = df.drop(['budget', 'revenue'], axis=1)
print(df)

# Converts the original languages into their own feature in boolean form
label_encoder = LabelEncoder()
df['lang_encoded'] = label_encoder.fit_transform(df['original_language'])
columns_to_view = ['original_language', 'lang_encoded']
df_copy = df[columns_to_view].copy()
print(df_copy)

# Create a mapping of each category to its encoded value
category_mapping = pd.DataFrame({
    'original_language': label_encoder.classes_,
    'Encoded': range(len(label_encoder.classes_))
})
language_dict = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
print(language_dict)




# Calculate company revenue based on the combined revenue of each of the companies working on the movie
# Split the cast column by ', ' and expand into a new row for each company
expanded_df = df.assign(production_companies=df['production_companies'].str.split(', ')).explode('production_companies')
# Aggregate the total revenue by the company
company_rev = expanded_df.groupby('production_companies')['Adjrevenue'].sum().reset_index()
# Rename columns for clarity
company_rev.rename(columns={'production_companies': 'company', 'Adjrevenue': 'company_rev'}, inplace=True)
# Sort by company revenue in descending order
company_rev = company_rev.sort_values(by='company_rev', ascending=False)
# Calculate total company_rev for each movie
# Merge company_rev with the original movie DataFrame
expanded_df = expanded_df.merge(company_rev, left_on='production_companies', right_on='company', how='left')
# Group by movie and sum the company_rev for each movie
movie_company_rev = expanded_df.groupby('id')['company_rev'].sum().reset_index()
# Merge movie company_rev back to the original DataFrame
df = df.merge(movie_company_rev, on='id', how='left')

# Drop original feature
df = df.drop(['original_language', 'production_companies', 'release_date'], axis=1)


# Function to process multi-word phrases (replaces spaces with underscores)
def process_keywords(keywords):
    # Split the keywords by commas
    keyword_list = keywords.split(", ")

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
columns_to_scale = ['vote_average', 'vote_count', 'runtime', 'popularity',
                    'AdjBudget', 'company_rev', 'Adjrevenue'
]
# Initialize the Min-Max Scaler
scaler = MinMaxScaler()
# Apply scaling to the selected columns
scaled_dataset[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
# Verify the scaling
print(scaled_dataset[columns_to_scale].describe())

# Merge scaled values back into the original dataset
df[columns_to_scale] = scaled_dataset[columns_to_scale]




# Preprocessed file is saved as Movies1M.csv
df.to_csv('Movies1Mscaled.csv', index=False)

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
#plt.show()






