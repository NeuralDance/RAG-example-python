
from dataloader.loader import get_embedding
from .knn_search import kNN_search
from .keyword_search import runBM25
from .rerank import rerank_rff
from .llm_generation import getLlmRespone, getTextForLlm


def RAG(query, data):
    # Perform kNN search
    input_embedding = get_embedding(query)
    top_3_embeddings = kNN_search(data, 'Embeddings', input_embedding)

    # perform BM25
    data = runBM25(data, query)

    # rerank with RFF
    data = rerank_rff(data)

    # get text for LLM based on top ranked Embeddings
    textForLlm = getTextForLlm(data, k=7)

    prompt = query + "Answer your question based on the following information: " + textForLlm

    response = getLlmRespone(prompt)

    return response
