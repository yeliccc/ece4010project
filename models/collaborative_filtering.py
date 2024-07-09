import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD


def cosine_similarity_manual(matrix):
    norms = np.linalg.norm(matrix, axis=1)
    similarity = np.dot(matrix, matrix.T) / (norms[:, None] * norms[None, :])
    return np.nan_to_num(similarity)


def normalize_ratings(matrix):
    means = matrix.mean(axis=1)
    normalized_matrix = matrix.subtract(means, axis=0)
    return normalized_matrix, means


def denormalize_ratings(predicted_matrix, means):
    return predicted_matrix.add(means, axis=0)


def user_based_recommendations(user_id, train_matrix, movies, num_recommendations=5, regularization=1e-5):
    try:
        train_matrix_filled = train_matrix.fillna(0).to_numpy()
        user_similarity = cosine_similarity_manual(train_matrix_filled)
        user_similarity_df = pd.DataFrame(user_similarity, index=train_matrix.index, columns=train_matrix.index)

        user_similarities = user_similarity_df.loc[user_id]
        similar_users = user_similarities.sort_values(ascending=False).index[1:]  # Exclude the user itself
        similar_user_ratings = train_matrix.loc[similar_users].dropna(thresh=1, axis=1)
        similar_user_ratings = similar_user_ratings.apply(lambda x: x * user_similarities[x.name], axis=1)
        weighted_average_ratings = similar_user_ratings.sum(axis=0) / (
                    user_similarities[similar_user_ratings.index].sum() + regularization)
        recommendations = weighted_average_ratings.sort_values(ascending=False).head(num_recommendations)

        recommendation_list = []
        for movie_id, score in recommendations.items():
            if movie_id in movies['movieId'].values:
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


def item_based_recommendations(user_id, train_matrix, movies, num_recommendations=5, regularization=1e-5):
    try:
        train_matrix_filled = train_matrix.fillna(0).T.to_numpy()
        item_similarity = cosine_similarity_manual(train_matrix_filled)
        item_similarity_df = pd.DataFrame(item_similarity, index=train_matrix.columns, columns=train_matrix.columns)

        user_ratings = train_matrix.loc[user_id].dropna()
        similar_scores = item_similarity_df[user_ratings.index].dot(user_ratings).div(
            item_similarity_df[user_ratings.index].sum(axis=1) + regularization)
        recommendations = similar_scores.sort_values(ascending=False).head(num_recommendations)

        recommendation_list = []
        for movie_id, score in recommendations.items():
            if movie_id in movies['movieId'].values:
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


def svd_matrix_factorization(train_matrix, n_components=20):
    matrix_filled = train_matrix.fillna(0)
    svd = TruncatedSVD(n_components=n_components)
    latent_matrix = svd.fit_transform(matrix_filled)
    return pd.DataFrame(latent_matrix, index=train_matrix.index,
                        columns=[f'feature_{i + 1}' for i in range(n_components)])
