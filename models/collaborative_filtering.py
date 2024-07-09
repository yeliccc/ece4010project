import pandas as pd
import numpy as np


def cosine_similarity_manual(matrix):
    norms = np.linalg.norm(matrix, axis=1)
    similarity = np.dot(matrix, matrix.T) / (norms[:, None] * norms[None, :])
    return np.nan_to_num(similarity)


def user_based_recommendations(user_id, train_matrix, movies, num_recommendations=5):
    try:
        train_matrix_filled = train_matrix.fillna(0).to_numpy()
        user_similarity = cosine_similarity_manual(train_matrix_filled)
        user_similarity_df = pd.DataFrame(user_similarity, index=train_matrix.index, columns=train_matrix.index)

        user_similarities = user_similarity_df.loc[user_id]
        similar_users = user_similarities.sort_values(ascending=False).index[1:]  # 排除自身
        similar_user_ratings = train_matrix.loc[similar_users].dropna(thresh=1, axis=1)
        similar_user_ratings = similar_user_ratings.apply(lambda x: x * user_similarities[x.name], axis=1).mean()
        recommendations = similar_user_ratings.sort_values(ascending=False).head(num_recommendations)

        recommendation_list = []
        for movie_id, score in recommendations.items():
            movie = movies[movies['movieId'] == movie_id].iloc[0]
            recommendation_list.append({
                'movieId': movie_id,
                'title': movie['title'],
                'genres': movie['genres'],
                'score': score
            })

        return recommendation_list
    except KeyError:
        raise ValueError(f"User ID {user_id} not found in training data")
    except Exception as e:
        raise RuntimeError(f"An error occurred during user-based recommendation: {e}")


def item_based_recommendations(user_id, train_matrix, movies, num_recommendations=5):
    try:
        train_matrix_filled = train_matrix.fillna(0).T.to_numpy()
        item_similarity = cosine_similarity_manual(train_matrix_filled)
        item_similarity_df = pd.DataFrame(item_similarity, index=train_matrix.columns, columns=train_matrix.columns)

        user_ratings = train_matrix.loc[user_id].dropna()
        similar_scores = item_similarity_df[user_ratings.index].dot(user_ratings).div(
            item_similarity_df[user_ratings.index].sum(axis=1))
        recommendations = similar_scores.sort_values(ascending=False).head(num_recommendations)

        recommendation_list = []
        for movie_id, score in recommendations.items():
            movie = movies[movies['movieId'] == movie_id].iloc[0]
            recommendation_list.append({
                'movieId': movie_id,
                'title': movie['title'],
                'genres': movie['genres'],
                'score': score
            })

        return recommendation_list
    except KeyError:
        raise ValueError(f"User ID {user_id} not found in training data")
    except Exception as e:
        raise RuntimeError(f"An error occurred during item-based recommendation: {e}")
