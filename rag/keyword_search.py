import pandas as pd
from rank_bm25 import BM25Okapi


# could be made a class Keywordsearch 
# Different types of keyword search could inherit (BM25, Splade, Stringsearch, ... ) from
def keywordSearch(df, query):

    # Tokenize the corpus
    tokenized_corpus = [doc.split(" ") for doc in df['TextChunk']]

    # Initialize the BM25 model
    bm25 = BM25Okapi(tokenized_corpus)

    # Define the query
    tokenized_query = query.split(" ")

    # Get the scores of the keyword search for each embedding in the table
    doc_scores = bm25.get_scores(tokenized_query)

    # Add BM25 scores to the embeddings table
    df['BM25_Score'] = doc_scores

    return df
