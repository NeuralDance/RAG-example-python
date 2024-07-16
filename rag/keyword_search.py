import pandas as pd
from rank_bm25 import BM25Okapi


def runBM25(df, query):

    # Tokenize the corpus
    tokenized_corpus = [doc.split(" ") for doc in df['TextChunk']]

    # Initialize the BM25 model
    bm25 = BM25Okapi(tokenized_corpus)

    # Define the query
    tokenized_query = query.split(" ")

    # Get the indices of the top N documents
    top_n_indices = bm25.get_top_n(tokenized_query, df['TextChunk'], n=10)

    doc_scores = bm25.get_scores(tokenized_query)

    # Use the indices to get the corresponding rows from the DataFrame
    top_n_rows = df.iloc[[df['TextChunk'].tolist().index(doc)
                          for doc in top_n_indices]]

    # Display the results
    top_n_rows

    df['BM25_Score'] = doc_scores

    return df
