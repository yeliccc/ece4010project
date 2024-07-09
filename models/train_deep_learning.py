import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 数据预处理
ratings = pd.read_csv('../ml-latest-small/ratings.csv')

user_ids = ratings['userId'].unique().tolist()
item_ids = ratings['movieId'].unique().tolist()

user_to_index = {x: i for i, x in enumerate(user_ids)}
item_to_index = {x: i for i, x in enumerate(item_ids)}

ratings['userId'] = ratings['userId'].apply(lambda x: user_to_index[x])
ratings['movieId'] = ratings['movieId'].apply(lambda x: item_to_index[x])

train, test = train_test_split(ratings, test_size=0.2, random_state=42)

# 标准化评分
scaler = StandardScaler()
train['rating'] = scaler.fit_transform(train['rating'].values.reshape(-1, 1))
test['rating'] = scaler.transform(test['rating'].values.reshape(-1, 1))

# 构建神经网络模型
n_users = len(user_ids)
n_items = len(item_ids)
n_factors = 50  # 潜在因子数量

user_input = Input(shape=(1,), name='user_input')
user_embedding = Embedding(input_dim=n_users, output_dim=n_factors, name='user_embedding')(user_input)
user_vec = Flatten()(user_embedding)
user_vec = Dropout(0.5)(user_vec)

item_input = Input(shape=(1,), name='item_input')
item_embedding = Embedding(input_dim=n_items, output_dim=n_factors, name='item_embedding')(item_input)
item_vec = Flatten()(item_embedding)
item_vec = Dropout(0.5)(item_vec)

concat = Concatenate()([user_vec, item_vec])
concat = Dense(128, activation='relu')(concat)
concat = BatchNormalization()(concat)
concat = Dropout(0.5)(concat)
concat = Dense(64, activation='relu')(concat)
concat = BatchNormalization()(concat)
concat = Dropout(0.5)(concat)
output = Dense(1)(concat)

model = Model(inputs=[user_input, item_input], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

# 训练模型
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

history = model.fit(
    [train['userId'], train['movieId']],
    train['rating'],
    epochs=20,
    verbose=1,
    validation_data=([test['userId'], test['movieId']], test['rating']),
    callbacks=[early_stopping, model_checkpoint]
)

# 评估模型并计算RMSE
model.load_weights('best_model.keras')
predictions = model.predict([test['userId'], test['movieId']])
mse = mean_squared_error(test['rating'], predictions)
rmse = np.sqrt(mse)

print(f'Test RMSE: {rmse}')

# 保存RMSE到文件
with open('deep_learning_rmse.txt', 'w') as f:
    f.write(f'{rmse}\n')
