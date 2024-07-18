import json
import pandas as pd
import matplotlib.pyplot as plt

class LogVisualizer:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.logs = self.load_logs()

    def load_logs(self):
        with open(self.log_file_path, 'r') as file:
            logs = json.load(file)
        return logs

    def visualize_user_queries(self):
        for entry in self.logs["user_query"]:
            #print(user_query)
            query = entry["query"]
            query_embedding = entry["input_embedding"]
        return query, query_embedding

    def visualize_keyword_search(self):
        records = []
        for entry in self.logs["rerank"]:
            for result in entry["reranked_results"]:
                record = {
                    "Filename": result.get("Filename"),
                    "TextChunk": result.get("TextChunk"),
                    "ChunkLength": result.get("ChunkLength"),
                    "PageNum": result.get("PageNum"),
                    "Embeddings": result.get("Embeddings"),
                    "BM25_Score": result.get("BM25_Score"),
                }
                records.append(record)
        data = pd.DataFrame(records).sort_values(by='BM25_Score', ascending=False)
        return data

    def visualize_semantic_search(self):
        records = []
        for entry in self.logs["rerank"]:
            for result in entry["reranked_results"]:
                record = {
                    "Filename": result.get("Filename"),
                    "TextChunk": result.get("TextChunk"),
                    "ChunkLength": result.get("ChunkLength"),
                    "PageNum": result.get("PageNum"),
                    "Embeddings": result.get("Embeddings"),
                    "kNN_distance": result.get("kNN_distance"),
                }
                records.append(record)
        data = pd.DataFrame(records).sort_values(by='kNN_distance', ascending=False)
        return data


    def visualize_rerank(self):
        records = []
        for entry in self.logs["rerank"]:
            for result in entry["reranked_results"]:
                record = {
                    "Filename": result.get("Filename"),
                    "TextChunk": result.get("TextChunk"),
                    "ChunkLength": result.get("ChunkLength"),
                    "PageNum": result.get("PageNum"),
                    "Embeddings": result.get("Embeddings"),
                    "kNN_distance": result.get("kNN_distance"),
                    "BM25_Score": result.get("BM25_Score"),
                    "rerank_rrf": result.get("rerank_rrf")
                }
                records.append(record)
        return pd.DataFrame(records)
        

    def visualize_llm_prompt(self):
        for entry in self.logs["llm_prompt"]:
            prompt = entry["prompt"]
        return prompt

    def visualize_final_llm_responses(self):
        for entry in self.logs["final_llm_response"]:
            response = entry["response"]
        return response



#log_file_path = "logs/86a8895c-f4bd-41a5-85a2-72d147b77119.json"  # Replace with actual path
#visualizer = LogVisualizer(log_file_path)

#visualizer.visualize_semantic_search()
#visualizer.visualize_keyword_search()
#visualizer.visualize_rerank()
#visualizer.visualize_user_queries()
#visualizer.visualize_final_llm_responses()
