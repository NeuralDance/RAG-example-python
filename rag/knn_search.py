import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def kNN_search(df, embeddings_column, input_embedding):

    k = len(df)

    # Convert the embeddings column to a numpy array
    embeddings = np.array(df[embeddings_column].tolist())

    # Compute cosine similarity between input_embedding and all embeddings
    similarities = cosine_similarity([input_embedding], embeddings)

    # Convert similarities to distances (since NearestNeighbors expects distances)
    distances = 1 - similarities[0]  # 1 - cosine_similarity gives us distances

    # Retrieve the indices sorted by increasing distance
    top_k_indices = np.argsort(distances)[:k]

    # Retrieve the top k embeddings and distances
    top_k_embeddings = df.iloc[top_k_indices]
    top_k_distances = distances[top_k_indices]

    # Add distances to the dataframe (if needed)
    df["kNN_distance"] = distances

    return top_k_embeddings.head(5)
