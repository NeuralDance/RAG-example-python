import pandas as pd
import numpy as np

# could be made a class rerank 
# Different types of reranking could inherit (rff, linear combination, cohere rerank , ... ) from
def rerank(df):
    
    # Initialize a dictionary to store reciprocal ranks
    rank_dict = {}

    # Iterate over each column (rankings from different sources)
    # search columns could be made a parameter of the reranking
    
    for col in ["BM25_Score", "kNN_distance"]:
        # Calculate reciprocal ranks
        ranks = df[col].rank(ascending=False, method='min')
        reciprocal_ranks = 1 / ranks
        rank_dict[col] = reciprocal_ranks

    # Concatenate reciprocal ranks into a new DataFrame
    merged_ranks = pd.DataFrame(rank_dict)

    # Calculate the sum of reciprocal ranks for each row
    df['rerank_rrf'] = merged_ranks.sum(axis=1)

    # Sort by the summed reciprocal ranks in descending order to order by relevance
    df = df.sort_values(by='rerank_rrf', ascending=False)

    return df
