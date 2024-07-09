from flask import Flask, request, render_template, flash, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mysql.connector
from utils.database import get_db_connection
from models.collaborative_filtering import item_based_recommendations, user_based_recommendations
from models.deep_learning_model import DeepLearningRecommender  # 新增

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# 初始化深度学习推荐模型
dl_recommender = DeepLearningRecommender()

# 读取深度学习模型的RMSE
with open('models/deep_learning_rmse.txt', 'r') as f:
    deep_learning_rmse = float(f.readline().strip())


def get_data_from_db():
    try:
        db = get_db_connection()
        cursor = db.cursor()
        cursor.execute("SELECT * FROM ratings")
        ratings = pd.DataFrame(cursor.fetchall(), columns=['userId', 'movieId', 'rating', 'timestamp'])
        cursor.execute("SELECT * FROM movies")
        movies = pd.DataFrame(cursor.fetchall(), columns=['movieId', 'title', 'genres'])
        cursor.close()
        db.close()
        return ratings, movies
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None, None


def calculate_rmse(predictions, actual):
    predictions = predictions[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return np.sqrt(mean_squared_error(predictions, actual))


def predict_ratings(matrix, similarity, type='user'):
    if type == 'user':
        return similarity.dot(matrix) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        return matrix.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])


# 初始加载数据和矩阵
ratings, movies = get_data_from_db()
if ratings is None or movies is None:
    raise ValueError("Failed to load data from the database")

dl_recommender.prepare_data(ratings)  # 准备深度学习推荐模型的数据

train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
train_matrix = train_data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
test_matrix = test_data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

user_similarity = cosine_similarity(train_matrix)
item_similarity = cosine_similarity(train_matrix.T)

user_prediction = predict_ratings(train_matrix.values, user_similarity, type='user')
item_prediction = predict_ratings(train_matrix.values, item_similarity, type='item')

user_rmse = calculate_rmse(user_prediction, test_matrix.values)
item_rmse = calculate_rmse(item_prediction, test_matrix.values)


@app.route('/')
def home():
    return render_template('index.html', user_rmse=user_rmse, item_rmse=item_rmse,
                           deep_learning_rmse=deep_learning_rmse)


@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        user_id = int(request.args.get('user_id'))
        num_recommendations = int(request.args.get('num_recommendations', 5))
        algorithm = request.args.get('algorithm')

        if user_id not in train_matrix.index:
            flash("User ID not found in the database", "danger")
            return redirect(url_for('home'))

        if algorithm == 'item_based':
            recommendations = item_based_recommendations(user_id, train_matrix, movies, num_recommendations)
        elif algorithm == 'user_based':
            recommendations = user_based_recommendations(user_id, train_matrix, movies, num_recommendations)
        elif algorithm == 'deep_learning':  # 新增
            recommended_items = dl_recommender.recommend(user_id, num_recommendations)
            recommendations = [{
                'movieId': item_id,
                'title': movies[movies['movieId'] == item_id]['title'].values[0],
                'genres': movies[movies['movieId'] == item_id]['genres'].values[0],
                'score': score
            } for item_id, score in recommended_items]
        else:
            recommendations = []
        return render_template('recommendations.html', recommendations=recommendations, user_id=user_id)
    except ValueError:
        flash("Invalid input. Please enter valid user ID and number of recommendations.", "danger")
        return redirect(url_for('home'))
    except Exception as e:
        flash(f"An error occurred: {e}", "danger")
        return redirect(url_for('home'))


@app.route('/new_user', methods=['GET', 'POST'])
def new_user():
    if request.method == 'POST':
        user_id = int(request.form.get('user_id'))

        # 检查user_id是否已存在
        global ratings, movies, train_matrix, user_similarity, item_similarity
        if user_id in ratings['userId'].values:
            flash("User ID already exists. Please choose a different User ID.", "danger")
            return redirect(url_for('new_user'))

        ratings_dict = {}
        for movie_id, rating in request.form.items():
            if movie_id.startswith('movie_'):
                movie_id = int(movie_id.split('_')[1])
                ratings_dict[movie_id] = float(rating)

        # 将新用户的评分添加到数据库中
        db = get_db_connection()
        cursor = db.cursor()
        for movie_id, rating in ratings_dict.items():
            cursor.execute("""
                INSERT INTO ratings (userId, movieId, rating, timestamp)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE rating=%s, timestamp=%s
            """, (user_id, movie_id, rating, int(pd.Timestamp.now().timestamp()), rating,
                  int(pd.Timestamp.now().timestamp())))
        db.commit()
        cursor.close()
        db.close()

        # 重新加载数据和更新矩阵
        ratings, movies = get_data_from_db()
        dl_recommender.prepare_data(ratings)
        train_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
        user_similarity = cosine_similarity(train_matrix)
        item_similarity = cosine_similarity(train_matrix.T)

        flash("Thank you for providing your ratings! Here are your recommendations.", "success")
        return redirect(url_for('recommend', user_id=user_id, num_recommendations=5, algorithm='user_based'))
    else:
        # 随机选择一些电影供新用户评分
        sample_movies = movies.sample(5).to_dict(orient='records')
        return render_template('new_user.html', sample_movies=sample_movies)


if __name__ == '__main__':
    app.run(debug=True)
