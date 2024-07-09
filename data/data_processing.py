import pandas as pd
import numpy as np
import mysql.connector
from sklearn.model_selection import train_test_split


def get_data_from_db():
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="yelicheng",
        database="movielens"
    )
    cursor = db.cursor()
    cursor.execute("SELECT * FROM ratings")
    ratings = pd.DataFrame(cursor.fetchall(), columns=['userId', 'movieId', 'rating', 'timestamp'])
    cursor.execute("SELECT * FROM movies")
    movies = pd.DataFrame(cursor.fetchall(), columns=['movieId', 'title', 'genres'])
    cursor.close()
    db.close()
    return ratings, movies


ratings, movies = get_data_from_db()
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
train_matrix = train_data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
test_matrix = test_data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
