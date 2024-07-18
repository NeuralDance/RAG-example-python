
from dataloader.loader import generateEmbedding
from .knn_search import calc_semantic_distances
from .keyword_search import keywordSearch
from .rerank import rerank
from .llm_generation import getLlmRespone, getTextForLlm
from .observe import Observer

def RAG(query, data,trace_tag):
    observer = Observer(trace_tag)

    # Generate embedding vector of the input user query
    input_embedding = generateEmbedding(query)
    observer.log_user_query(query,input_embedding)

    # Semantic Search: perform kNN search with cosine similarity
    data["kNN_distance"] = calc_semantic_distances(data, 'Embeddings', input_embedding)# chnge to semanti csesarch 
    observer.log_semantic_search(data["kNN_distance"])

    # Keyword Search: perform BM25
    data['BM25_Score'] = keywordSearch(data, query)
    observer.log_keyword_search(data['BM25_Score'])   

    # Rerank: rerank with RFF
    data = rerank(data)
    observer.log_rerank(data[0:2]) 

    # get text for LLM based on top ranked Embeddings by RAG
    retrievedEmbeddingTexts = getTextForLlm(data, k=7)

    # Prepare user prompt based on user input and retrieved text embeddings by RAG
    prompt = query + "Answer your question based on the following information: " + retrievedEmbeddingTexts
    observer.log_llm_prompt(prompt)

    # Generate an LLM response
    response = getLlmRespone(prompt)
    observer.log_final_llm_response(response)

    # Save the logs in JSON format
    observer.save_logs_to_file("rag/observe/logs")

    return response
