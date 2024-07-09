import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.database import get_db_connection

def get_data():
    ratings, movies = load_data_from_db()
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
    train_matrix = train_data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    test_matrix = test_data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    return ratings, movies, train_matrix, test_matrix

def load_data_from_db():
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM ratings")
    ratings = pd.DataFrame(cursor.fetchall(), columns=['userId', 'movieId', 'rating', 'timestamp'])
    cursor.execute("SELECT * FROM movies")
    movies = pd.DataFrame(cursor.fetchall(), columns=['movieId', 'title', 'genres'])
    cursor.close()
    db.close()
    return ratings, movies
