import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, KNNBasic, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse
from sklearn.metrics import ndcg_score, f1_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import streamlit as st

ratings = pd.read_csv("Ratings.csv")
books = pd.read_csv("Books.csv", low_memory=False)
users = pd.read_csv("Users.csv")

ratings_clean = ratings[ratings['Book-Rating'] > 0]
active_users = ratings_clean['User-ID'].value_counts()
active_users = active_users[active_users >= 50].index
ratings_clean = ratings_clean[ratings_clean['User-ID'].isin(active_users)]

reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings_clean[['User-ID', 'ISBN', 'Book-Rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model_knn = KNNBasic()
model_knn.fit(trainset)

model_svd = SVD()
model_svd.fit(trainset)

df_ml = ratings_clean.merge(books[['ISBN', 'Book-Title']], on='ISBN', how='left')
df_ml['user_enc'] = LabelEncoder().fit_transform(df_ml['User-ID'])
df_ml['book_enc'] = LabelEncoder().fit_transform(df_ml['ISBN'])

X = df_ml[['user_enc', 'book_enc']]
y = df_ml['Book-Rating']
X_train_ml, X_test_ml, y_train_ml, y_test_ml = sk_train_test_split(X, y, test_size=0.2, random_state=42)

dtree = DecisionTreeRegressor(max_depth=10)
dtree.fit(X_train_ml, y_train_ml)

user_id_map = dict(zip(df_ml['User-ID'], df_ml['user_enc']))
book_id_map = dict(zip(df_ml['ISBN'], df_ml['book_enc']))
book_title_map = dict(zip(books['Book-Title'].str.lower(), books['ISBN']))
book_info_map = dict(zip(books['ISBN'], books['Book-Title']))

def get_recommendations(user_id=None, book_title=None, model_type="SVD", top_n=5):
    model_type = model_type.upper()
    used_isbns = ratings_clean['ISBN'].unique()

    if book_title:
        book_title = book_title.lower().strip()
        isbn = book_title_map.get(book_title)
        if not isbn:
            return f"Book '{book_title}' not found."

    if model_type == "SVD":
        model = model_svd
    elif model_type == "KNN":
        model = model_knn
    elif model_type == "DT":
        model = dtree
    else:
        return "Invalid model type. Choose from SVD, KNN, or DT."

    if user_id and model_type in ["SVD", "KNN"]:
        rated_books = ratings_clean[ratings_clean['User-ID'] == user_id]['ISBN'].tolist()
        candidate_books = [isbn for isbn in used_isbns if isbn not in rated_books]
        preds = [model.predict(user_id, isbn) for isbn in candidate_books]
        top_preds = sorted(preds, key=lambda x: x.est, reverse=True)[:top_n]
        return [(book_info_map[p.iid], round(p.est, 2)) for p in top_preds]

    elif user_id and model_type == "DT":
        if user_id not in user_id_map:
            return "User not found."
        user_enc = user_id_map[user_id]
        candidate_books = [isbn for isbn in used_isbns if isbn in book_id_map]
        book_encs = [book_id_map[isbn] for isbn in candidate_books]
        X_pred = pd.DataFrame({'user_enc': [user_enc]*len(book_encs), 'book_enc': book_encs})
        preds = dtree.predict(X_pred)
        results = pd.DataFrame({'ISBN': candidate_books, 'Rating': preds})
        results = results.sort_values(by='Rating', ascending=False).head(top_n)
        return [(book_info_map[isbn], round(rating, 2)) for isbn, rating in zip(results['ISBN'], results['Rating'])]

    elif book_title:
        book_data = books[books['Book-Title'].str.lower() == book_title]
        if book_data.empty:
            return "Book not found."
        author = book_data.iloc[0]['Book-Author']
        publisher = book_data.iloc[0]['Publisher']
        similar_books = books[(books['Book-Author'] == author) | (books['Publisher'] == publisher)]
        rated = ratings_clean.groupby('ISBN')['Book-Rating'].mean().reset_index()
        ranked_books = similar_books.merge(rated, on='ISBN', how='left').sort_values(by='Book-Rating', ascending=False)
        return list(ranked_books[['Book-Title', 'Book-Rating']].dropna().head(top_n).itertuples(index=False, name=None))

    return "Please provide at least a user_id or book title."

st.title("📚BookGenie🧞‍♂️")
st.write("Provide a User ID and/or Book Title to get recommendations.")

user_id_input = st.text_input("Enter User ID (optional):")
book_title_input = st.text_input("Enter Book Title (optional):")
model_type_input = st.selectbox("Choose Model", ["SVD", "KNN", "DT"])

if st.button("Get Recommendations"):
    if not user_id_input and not book_title_input:
        st.warning("Please provide at least a User ID or Book Title.")  
    else:
        try:
            user_id_val = int(user_id_input) if user_id_input else None  
            result = get_recommendations(user_id=user_id_val, book_title=book_title_input, model_type=model_type_input)  
            st.subheader("Top Recommendations:")
            if isinstance(result, str):
                st.error(result)  
            else:
                if book_title_input and not user_id_input:
                    for title, score in result:
                        st.markdown(f"**{title}** — Average User Rating: {score}")
                else:
                    for title, score in result:
                        st.markdown(f"**{title}** — Predicted Rating: {score}") 
        except Exception as e:
            st.error(f"Error: {e}")
