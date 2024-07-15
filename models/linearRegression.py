import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib  # 用于保存和加载模型

# Load the CSV file
file_path = '../ml-latest-small/ratings.csv'
ratings_df = pd.read_csv(file_path)

# Split the data into training and testing sets
train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Create user and item feature vectors
user_ids = ratings_df['userId'].unique()
movie_ids = ratings_df['movieId'].unique()

user_id_map = {user_id: i for i, user_id in enumerate(user_ids)}
movie_id_map = {movie_id: i for i, movie_id in enumerate(movie_ids)}

train_data['user_id'] = train_data['userId'].map(user_id_map)
train_data['movie_id'] = train_data['movieId'].map(movie_id_map)
test_data['user_id'] = test_data['userId'].map(user_id_map)
test_data['movie_id'] = test_data['movieId'].map(movie_id_map)

# Prepare training data
X_train = train_data[['user_id', 'movie_id']]
y_train = train_data['rating']

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'linear_regression_model.pkl')
joblib.dump(user_id_map, 'user_id_map.pkl')
joblib.dump(movie_id_map, 'movie_id_map.pkl')

# Prepare testing data
X_test = test_data[['user_id', 'movie_id']]
y_test = test_data['rating']

# Predict ratings
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Linear Regression RMSE: {rmse}')
