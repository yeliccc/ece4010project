import pandas as pd
import mysql.connector

# 加载数据
ratings = pd.read_csv('../ml-latest-small/ratings.csv')
movies = pd.read_csv('../ml-latest-small/movies.csv')

# 连接到MySQL数据库
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="yelicheng",
    database="movielens"
)
cursor = db.cursor()

# 插入movies数据
for index, row in movies.iterrows():
    cursor.execute("""
        INSERT INTO movies (movieId, title, genres)
        VALUES (%s, %s, %s)
    """, (row['movieId'], row['title'], row['genres']))
db.commit()

# 插入ratings数据
for index, row in ratings.iterrows():
    cursor.execute("""
        INSERT INTO ratings (userId, movieId, rating, timestamp)
        VALUES (%s, %s, %s, %s)
    """, (row['userId'], row['movieId'], row['rating'], row['timestamp']))
db.commit()

# 关闭数据库连接
cursor.close()
db.close()
