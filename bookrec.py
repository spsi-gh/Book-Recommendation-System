import pandas as pd  # Used for reading and manipulating tabular data like reading csv files,filtering data,merging datasets
import matplotlib.pyplot as plt  # Used for plotting graphs and visualizations like bar plots and scatter plots
import seaborn as sns  # Used for making statistical visualizations prettier,basically is built on top of matplotlib
from surprise import Dataset, Reader, KNNBasic, SVD  # surprise library for collaborative filtering models
from surprise.model_selection import train_test_split  # To split the dataset into training and testing for surprise
from surprise.accuracy import rmse  # To calculate RMSE (Root Mean Squared Error) for model evaluation
from sklearn.metrics import ndcg_score, f1_score, mean_squared_error  # ML evaluation metrics
from sklearn.tree import DecisionTreeRegressor  # Traditional ML model for regression
from sklearn.model_selection import train_test_split as sk_train_test_split  # Sklearn's version of train/test split
from sklearn.preprocessing import LabelEncoder  # Encodes user and book IDs into numbers
import numpy as np  # Numerical operations
import streamlit as st  # Used for building the web interface

# Load datasets
ratings = pd.read_csv("Ratings.csv")  # Load ratings data into DataFrame
books = pd.read_csv("Books.csv", low_memory=False)  # Load book metadata
users = pd.read_csv("Users.csv")  # Load user data

# Filter out rows with 0 rating (not useful for training)
ratings_clean = ratings[ratings['Book-Rating'] > 0]

# Count how many books each user has rated
active_users = ratings_clean['User-ID'].value_counts()

# Keep only users who have rated 50 or more books
active_users = active_users[active_users >= 50].index

# Filter ratings for only active users
ratings_clean = ratings_clean[ratings_clean['User-ID'].isin(active_users)]

# Create reader object with rating scale for Surprise
reader = Reader(rating_scale=(1, 10))

# Load ratings into Surprise's dataset format
data = Dataset.load_from_df(ratings_clean[['User-ID', 'ISBN', 'Book-Rating']], reader)

# Split the Surprise dataset into train and test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Initialize and train KNN-based collaborative filtering model
model_knn = KNNBasic()
model_knn.fit(trainset)

# Initialize and train SVD-based collaborative filtering model
model_svd = SVD()
model_svd.fit(trainset)

# Merge ratings with book titles for encoding and later use
df_ml = ratings_clean.merge(books[['ISBN', 'Book-Title']], on='ISBN', how='left')

# Encode user IDs as numbers for traditional ML
df_ml['user_enc'] = LabelEncoder().fit_transform(df_ml['User-ID'])

# Encode book ISBNs as numbers for traditional ML
df_ml['book_enc'] = LabelEncoder().fit_transform(df_ml['ISBN'])

# Define input features and target label
X = df_ml[['user_enc', 'book_enc']]  # Features: user and book codes
y = df_ml['Book-Rating']  # Target: rating given

# Split the ML data into train and test sets
X_train_ml, X_test_ml, y_train_ml, y_test_ml = sk_train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Decision Tree Regressor
dtree = DecisionTreeRegressor(max_depth=10)
dtree.fit(X_train_ml, y_train_ml)

# Create lookup dictionary to map original user IDs to encoded user IDs
user_id_map = dict(zip(df_ml['User-ID'], df_ml['user_enc']))

# Create lookup dictionary to map original ISBNs to encoded book IDs
book_id_map = dict(zip(df_ml['ISBN'], df_ml['book_enc']))

# Map book titles to ISBNs (used in content-based filtering)
book_title_map = dict(zip(books['Book-Title'].str.lower(), books['ISBN']))

# Map ISBNs to book titles (used for display)
book_info_map = dict(zip(books['ISBN'], books['Book-Title']))

# Main function to generate recommendations
def get_recommendations(user_id=None, book_title=None, model_type="SVD", top_n=5):
    model_type = model_type.upper()  # Normalize model name to uppercase
    used_isbns = ratings_clean['ISBN'].unique()  # List of unique books in dataset

    if book_title:
        book_title = book_title.lower().strip()  # Normalize book title
        isbn = book_title_map.get(book_title)  # Lookup ISBN by title
        if not isbn:
            return f"Book '{book_title}' not found."

    # Choose model based on input
    if model_type == "SVD":
        model = model_svd
    elif model_type == "KNN":
        model = model_knn
    elif model_type == "DT":
        model = dtree
    else:
        return "Invalid model type. Choose from SVD, KNN, or DT."

    # Collaborative Filtering for SVD and KNN
    if user_id and model_type in ["SVD", "KNN"]:
        rated_books = ratings_clean[ratings_clean['User-ID'] == user_id]['ISBN'].tolist()  # Books user already rated
        candidate_books = [isbn for isbn in used_isbns if isbn not in rated_books]  # Books not rated yet
        preds = [model.predict(user_id, isbn) for isbn in candidate_books]  # Predict ratings for unseen books
        top_preds = sorted(preds, key=lambda x: x.est, reverse=True)[:top_n]  # Top predictions sorted by rating
        return [(book_info_map[p.iid], round(p.est, 2)) for p in top_preds]  # Return top recommendations

    # Decision Tree model prediction
    elif user_id and model_type == "DT":
        if user_id not in user_id_map:
            return "User not found."
        user_enc = user_id_map[user_id]  # Encode user ID
        candidate_books = [isbn for isbn in used_isbns if isbn in book_id_map]  # Valid books
        book_encs = [book_id_map[isbn] for isbn in candidate_books]  # Encode books
        X_pred = pd.DataFrame({'user_enc': [user_enc]*len(book_encs), 'book_enc': book_encs})  # Input for prediction
        preds = dtree.predict(X_pred)  # Predict ratings
        results = pd.DataFrame({'ISBN': candidate_books, 'Rating': preds})  # Combine predictions with ISBNs
        results = results.sort_values(by='Rating', ascending=False).head(top_n)  # Top results
        return [(book_info_map[isbn], round(rating, 2)) for isbn, rating in zip(results['ISBN'], results['Rating'])]

    # Content-Based Filtering
    elif book_title:
        book_data = books[books['Book-Title'].str.lower() == book_title]  # Match book title
        if book_data.empty:
            return "Book not found."
        author = book_data.iloc[0]['Book-Author']  # Get author of the book
        publisher = book_data.iloc[0]['Publisher']  # Get publisher
        similar_books = books[(books['Book-Author'] == author) | (books['Publisher'] == publisher)]  # Find similar books
        rated = ratings_clean.groupby('ISBN')['Book-Rating'].mean().reset_index()  # Average rating per book
        ranked_books = similar_books.merge(rated, on='ISBN', how='left').sort_values(by='Book-Rating', ascending=False)  # Merge and sort
        return list(ranked_books[['Book-Title', 'Book-Rating']].dropna().head(top_n).itertuples(index=False, name=None))  # Return top books

    return "Please provide at least a user_id or book title."  # Fallback message if no input

# Streamlit UI starts here
st.title("📚 Book Recommendation System")  # Web app title
st.write("Provide a User ID and/or Book Title to get recommendations.")  # Instruction message

# Text input for user ID and book title
user_id_input = st.text_input("Enter User ID (optional):")
book_title_input = st.text_input("Enter Book Title (optional):")
model_type_input = st.selectbox("Choose Model", ["SVD", "KNN", "DT"])  # Model selection dropdown

# Button to get recommendations
if st.button("Get Recommendations"):
    if not user_id_input and not book_title_input:
        st.warning("Please provide at least a User ID or Book Title.")  # Warn if no input
    else:
        try:
            user_id_val = int(user_id_input) if user_id_input else None  # Convert user ID to int
            result = get_recommendations(user_id=user_id_val, book_title=book_title_input, model_type=model_type_input)  # Call function
            st.subheader("Top Recommendations:")
            if isinstance(result, str):
                st.error(result)  # Display error if result is a string
            else:
                for title, score in result:
                    st.markdown(f"*{title}* — Predicted Rating: {score}")  # Display each recommendation
        except Exception as e:
            st.error(f"Error: {e}")  # Handle any runtime errors 