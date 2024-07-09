import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import os


class DeepLearningRecommender:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), 'best_model.keras')
        if not os.path.exists(model_path):
            raise ValueError(
                f"File not found: filepath={model_path}. Please ensure the file is an accessible `.keras` file.")
        self.model = load_model(model_path)
        self.user_to_index = {}
        self.item_to_index = {}
        self.index_to_user = {}
        self.index_to_item = {}
        self.scaler = StandardScaler()  # 保存标准化器

    def prepare_data(self, ratings):
        self.user_to_index = {x: i for i, x in enumerate(ratings['userId'].unique().tolist())}
        self.item_to_index = {x: i for i, x in enumerate(ratings['movieId'].unique().tolist())}
        self.index_to_user = {i: x for x, i in self.user_to_index.items()}
        self.index_to_item = {i: x for x, i in self.item_to_index.items()}
        self.scaler.fit(ratings['rating'].values.reshape(-1, 1))  # 训练标准化器

    def recommend(self, user_id, num_recommendations=5):
        if user_id not in self.user_to_index:
            return []
        user_idx = self.user_to_index[user_id]
        user_array = np.array([user_idx] * len(self.item_to_index))
        item_array = np.array(list(self.item_to_index.values()))

        predictions = self.model.predict([user_array, item_array]).flatten()
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()  # 逆标准化
        top_indices = predictions.argsort()[-num_recommendations:][::-1]
        recommended_items = [(self.index_to_item[idx], predictions[idx]) for idx in top_indices]

        return recommended_items
