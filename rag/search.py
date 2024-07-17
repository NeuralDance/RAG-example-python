
from dataloader.loader import generateEmbedding
from .knn_search import semanticSearch
from .keyword_search import keywordSearch
from .rerank import rerank
from .llm_generation import getLlmRespone, getTextForLlm


def RAG(query, data):
    # Generate embedding vector of the input user query
    input_embedding = generateEmbedding(query)

    # Semantic Search: perform kNN search with cosine similarity
    data = semanticSearch(data, 'Embeddings', input_embedding)

    # Keyword Search: perform BM25
    data = keywordSearch(data, query)

    # Rerank: rerank with RFF
    data = rerank(data)

    # get text for LLM based on top ranked Embeddings by RAG
    retrievedEmbeddingTexts = getTextForLlm(data, k=7)

    # Prepare user prompt based on user input and retrieved text embeddings by RAG
    prompt = query + "Answer your question based on the following information: " + retrievedEmbeddingTexts

    # Generate an LLM response
    response = getLlmRespone(prompt)

    return response
