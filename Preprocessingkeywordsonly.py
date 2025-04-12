import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import time
from sklearn.feature_extraction.text import TfidfVectorizer

# Takes a while to run so this just checks
start_time = time.time()
df = pd.read_csv('TMDB_movie_dataset_v11.csv', encoding='ISO-8859-1')

#removes all adult movies
df = df[~df['adult']]
df = df[~((df["vote_average"] == 0) & (df["vote_count"] == 0))]
df = df[~((df["revenue"] == 1))]

status_counts_before = df['status'].value_counts()
print("Status counts before filtering:\n", status_counts_before, "\n")

df = df[df['status'] == 'Released']
# "id","title","vote_average","vote_count","status","release_date","revenue","runtime","adult","backdrop_path","budget","homepage",
# "imdb_id","original_language","original_title","overview","popularity","poster_path","tagline","genres","production_companies",
# "production_countries","spoken_languages","keywords"
# dropping irrelevant columns
features_to_drop = ["id","title","vote_average","vote_count","status","runtime","adult","backdrop_path","budget","homepage","imdb_id","original_language","original_title","overview","popularity","poster_path","tagline","production_companies","production_countries","spoken_languages"]
df = df.drop(features_to_drop, axis=1)


# Checks if there are missing values that could cause issues later on
missing_values = df.isnull().sum()
# print(missing_values)
# drop rows with missing values
df = df.dropna()
# check for missing values again to confirm
# missing_values = df.isnull().sum()
df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
df = df[df["revenue"] != 0]

# print(df["revenue"].value_counts())

# Convert release_date to datetime format
df['release_date'] = pd.to_datetime(df['release_date'])
# Extract release year
df['release_year'] = df['release_date'].dt.year



df = df[(df['release_year'] >= 1913) & (df['release_year'] <= 2024)]
# min_year = df['release_year'].min()
# max_year = df['release_year'].max()

df = df.sort_values(by="release_year", ascending=False).head(2000)


df = df.drop(['release_date'], axis=1)
print(df.shape)
print(df.head)


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



# Function to process multi-word phrases (replaces spaces with underscores)
def process_keywords(keywords):
    if pd.isna(keywords):  # Handle NaN values
        return []

    # Split the keywords by commas
    keyword_list = keywords.split(", ")

    processed_keywords = []
    for kw in keyword_list:
        kw = kw.encode("utf-8", "ignore").decode("utf-8")  # Remove non-UTF characters
        kw = kw.encode("ascii", "ignore").decode("ascii")  # Remove non-ASCII characters
        # kw = kw.lower()  # Convert to lowercase
        kw = "".join(c if c.isalnum() or c.isspace() else " " for c in kw)  # Remove special chars
        kw = kw.replace(" ", "_")  # Replace spaces in multi-word phrases with "_"
        processed_keywords.append(kw)

    print(keyword_list)

    return processed_keywords


# Apply the function to the keywords column
# Process keywords column
df["keywords"] = df["keywords"].apply(process_keywords)

# Convert list of keywords into a single string per row
df["keywords"] = df["keywords"].apply(lambda x: " ".join(x))
df.to_csv('notinuse.csv', index=False)
df = df.reset_index(drop=True)

# Apply TF-IDF transformation
tfidf = TfidfVectorizer(max_features=500, stop_words="english",
                        analyzer="word")  # token_pattern ensures that words are captured properly
tfidf_matrix = tfidf.fit_transform(df["keywords"])

# Convert TF-IDF matrix to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

# Concatenate TF-IDF features with the original dataset
df = pd.concat([df.drop(columns=["keywords"]), tfidf_df], axis=1)  # Drop original keywords column

# Show completion message and new shape
print("TF-IDF preprocessing completed. New shape:", df.shape)
print(df.head())
print(df.head(10))
print(df.shape)







# Preprocessed file is saved as Movies1M.csv
df.to_csv('Movies1Mkeywords.csv', index=False)

# End timer
end_time = time.time()

# Calculate time taken to run
elapsed_time = end_time - start_time
print("Time taken: ", elapsed_time)







