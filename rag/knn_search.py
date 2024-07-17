import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# could be made a class semanticSearch 
# Different types of semantic search could inherit (kNN, aNN, ... ) from
def semanticSearch(df, embeddings_column, input_embedding):

    # setting k equal to the number of embeddings in the table 
    # to retrieve similarity metrics for all text chunks
    k = len(df) 

    # Convert the embeddings column to a numpy array
    embeddings = np.array(df[embeddings_column].tolist())

    # Compute cosine similarity between input_embedding and all embeddings
    # this could also be L2 distance etc. cosine is most popular in NLP
    similarities = cosine_similarity([input_embedding], embeddings)

    # Convert similarities to distances (since NearestNeighbors expects distances)
    distances = 1 - similarities[0]  # 1 - cosine_similarity gives us distances

    # Add kNN scores to the embeddings table
    df["kNN_distance"] = distances

    return df
