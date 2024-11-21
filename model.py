import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle

# Load the data (adjust path as needed)
books = pd.read_csv('Books.csv')
users = pd.read_csv('users.csv')
ratings = pd.read_csv('ratings.csv')

# Data preprocessing (same as your original code)
books.rename(columns={
    "Book-Title":"title",
    "Book-Author":"author",
    "Year-Of-Publication": "year",
    "Publisher":"publisher",
    "Image-URL-S":"img_url"
}, inplace=True)

ratings.rename(columns={
    "User-ID": "user_id",
    "Book-Rating": "rating"}, inplace=True)

x = ratings['user_id'].value_counts() > 100
y = x[x].index

ratings = ratings[ratings['user_id'].isin(y)]
rating_with_books = ratings.merge(books, on="ISBN")

num_rating = rating_with_books.groupby('title')['rating'].count().reset_index()
num_rating.rename(columns={"rating": "num_of_rating"}, inplace=True)

final_rating = rating_with_books.merge(num_rating, on='title')
final_rating = final_rating[final_rating['num_of_rating'] >= 50]

final_rating.drop_duplicates(['user_id', 'title'], inplace=True)

book_pivot = final_rating.pivot_table(columns='user_id', index='title', values='rating')
book_pivot.fillna(0, inplace=True)

book_sparse = csr_matrix(book_pivot)

model = NearestNeighbors(algorithm='brute')
model.fit(book_sparse)

# Save the model (optional, used for Flask later)
pickle.dump(model, open('artifacts/model.pkl', 'wb'))
pickle.dump(book_pivot.index, open('artifacts/books_name.pkl', 'wb'))
pickle.dump(final_rating, open('artifacts/final_rating.pkl', 'wb'))
pickle.dump(book_pivot, open('artifacts/book_pivot.pkl', 'wb'))

# Test the recommendation function
def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    
    recommended_books = []
    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        recommended_books.extend(books)  # Add recommended books to the list
    
    return recommended_books

# Test the function with an example book title
book_name = "1984"  # Replace with an existing book title from your dataset
recommended_books = recommend_book(book_name)

# Print the recommendations
print(f"Recommended books for '{book_name}':")
for book in recommended_books:
    print(book)
