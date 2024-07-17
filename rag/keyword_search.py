import pandas as pd
from rank_bm25 import BM25Okapi
from typing import List


# could be made a class Keywordsearch 
# Different types of keyword search could inherit (BM25, Splade, Stringsearch, ... ) from
def keywordSearch(df: pd.DataFrame, query: str) -> List[float]:

    # Tokenize the corpus
    tokenized_corpus = [doc.split(" ") for doc in df['TextChunk']]

    # Initialize the BM25 model
    bm25 = BM25Okapi(tokenized_corpus)

    # Define the query
    tokenized_query = query.split(" ")

    # Get the scores of the keyword search for each embedding in the table
    doc_scores = bm25.get_scores(tokenized_query)

    return doc_scores