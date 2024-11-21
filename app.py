from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle

app = Flask(__name__)

# Load pre-trained model and data
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))
books_name = pickle.load(open('artifacts/books_name.pkl', 'rb'))

@app.route('/recommend', methods=['GET'])
def recommend():
    book_name = request.args.get('book_name')
    
    # Check if the book title exists in the pivot table index
    if book_name not in book_pivot.index:
        return jsonify({"error": f"'{book_name}' not found in the dataset."}), 404

    # If the book exists, proceed with recommendations
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6)
    
    recommended_books = []
    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            recommended_books.append(j)
    
    return jsonify(recommended_books)

if __name__ == '__main__':
    app.run(debug=True)
