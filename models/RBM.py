import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


class RBM:
    def __init__(self, visible_units, hidden_units, learning_rate=0.01, epochs=50, batch_size=10, regularization=0.01):
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization = regularization

        # 初始化权重和偏置
        self.weights = tf.Variable(tf.random.normal([self.visible_units, self.hidden_units], stddev=0.01, dtype=tf.float32))
        self.visible_bias = tf.Variable(tf.zeros([self.visible_units], dtype=tf.float32))
        self.hidden_bias = tf.Variable(tf.zeros([self.hidden_units], dtype=tf.float32))

    def sigmoid(self, x):
        return tf.nn.sigmoid(x)

    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs), dtype=tf.float32)))

    def train(self, data):
        num_samples = data.shape[0]

        for epoch in range(self.epochs):
            np.random.shuffle(data)
            for batch_start in range(0, num_samples, self.batch_size):
                batch = data[batch_start:batch_start + self.batch_size].astype(np.float32)

                with tf.GradientTape() as tape:
                    # 正向传播：计算隐层节点的激活概率和状态
                    positive_hidden_probs = self.sigmoid(tf.matmul(batch, self.weights) + self.hidden_bias)
                    positive_hidden_states = self.sample_prob(positive_hidden_probs)

                    # 反向传播：重构可见层，计算隐层节点的重构概率
                    negative_visible_probs = self.sigmoid(
                        tf.matmul(positive_hidden_states, self.weights, transpose_b=True) + self.visible_bias)
                    negative_hidden_probs = self.sigmoid(
                        tf.matmul(negative_visible_probs, self.weights) + self.hidden_bias)

                    positive_assoc = tf.matmul(tf.transpose(batch), positive_hidden_probs)
                    negative_assoc = tf.matmul(tf.transpose(negative_visible_probs), negative_hidden_probs)

                    # 计算重构误差和正则化项
                    reconstruction_loss = tf.reduce_mean(tf.square(batch - negative_visible_probs))
                    regularization_loss = self.regularization * tf.reduce_sum(tf.square(self.weights))
                    cost = reconstruction_loss + regularization_loss

                # 更新权重和偏置
                gradients = tape.gradient(cost, [self.weights, self.visible_bias, self.hidden_bias])
                optimizer.apply_gradients(zip(gradients, [self.weights, self.visible_bias, self.hidden_bias]))

            print(f"Epoch {epoch + 1}/{self.epochs} completed. Cost: {cost.numpy()}")

    def predict(self, data):
        hidden_probs = self.sigmoid(tf.matmul(data.astype(np.float32), self.weights) + self.hidden_bias)
        visible_probs = self.sigmoid(tf.matmul(hidden_probs, self.weights, transpose_b=True) + self.visible_bias)
        return visible_probs


# 加载数据
data = pd.read_csv('../ml-latest-small/ratings.csv')

# 准备数据
data_matrix = data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
scaler = MinMaxScaler()
data_matrix = scaler.fit_transform(data_matrix).astype(np.float32)

# 初始化和训练RBM
visible_units = data_matrix.shape[1]
hidden_units = 64  # 隐层节点数量可调整
rbm = RBM(visible_units, hidden_units, learning_rate=0.01, epochs=10, batch_size=10)

optimizer = tf.optimizers.Adam(learning_rate=rbm.learning_rate)
rbm.train(data_matrix)


# 进行预测并计算RMSE
def calculate_rmse(rbm, data_matrix, scaler):
    predictions = []
    true_ratings = []

    for user_id in range(data_matrix.shape[0]):
        user_data = data_matrix[user_id].reshape(1, -1)
        prediction = rbm.predict(user_data)
        prediction = scaler.inverse_transform(prediction)

        true_data = scaler.inverse_transform(user_data)
        predictions.extend(prediction.flatten())
        true_ratings.extend(true_data.flatten())

    rmse = np.sqrt(mean_squared_error(true_ratings, predictions))
    return rmse


# 计算并输出RMSE
rmse = calculate_rmse(rbm, data_matrix, scaler)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
