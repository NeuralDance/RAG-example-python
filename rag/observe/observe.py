import json
import uuid
from datetime import datetime
import pandas as pd
import os 

class Observer:
    def __init__(self,trace_tag=None):
        self.logs = {
            "trace_tag": trace_tag,
            "trace_id": str(uuid.uuid4()),  # Generate a unique trace_id
            "timestamp": datetime.now().isoformat(),
            "semantic_search": [],
            "keyword_search": [],
            "rerank": [],
            "user_query": [],
            "final_llm_response": [],
            "llm_prompt":[],
            "final_llm_response":[],
        }

    def _convert_to_serializable(self, data):
        if isinstance(data, pd.Series):
            return data.tolist()
        elif isinstance(data, pd.DataFrame):
            return data.to_dict(orient='records')
        return data

    def log_semantic_search(self, results):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "results": self._convert_to_serializable(results)
        }
        self.logs["semantic_search"].append(log_entry)

    def log_keyword_search(self, results):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "results": self._convert_to_serializable(results)
        }
        self.logs["keyword_search"].append(log_entry)

    def log_rerank(self, reranked_results):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "reranked_results": self._convert_to_serializable(reranked_results)
        }
        self.logs["rerank"].append(log_entry)

    def log_user_query(self, query,input_embedding):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "input_embedding": self._convert_to_serializable(input_embedding),

        }
        self.logs["user_query"].append(log_entry)
    
    def log_llm_prompt(self, prompt):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
        }
        self.logs["llm_prompt"].append(log_entry)

    def log_final_llm_response(self, response):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "response": self._convert_to_serializable(response)
        }
        self.logs["final_llm_response"].append(log_entry)

    def get_logs_as_json(self):
        return json.dumps(self.logs, indent=4)

    def save_logs_to_file(self, directory):
        """
        : write logs to json. If no trace_tag is provided, take session_if as filename. 
        If trace_tag is provided check if trace_tag already exist and if so ask user to provide new one.
        """
        os.makedirs(directory, exist_ok=True)
        if self.logs['trace_tag'] is not None:
            filename = f"{directory}/{self.logs['trace_tag']}.json"
        else:
            filename = f"{directory}/{self.logs['session_id']}.json"
        
        if os.path.exists(filename):
            new_tag = input(f"A file with tag '{self.logs['trace_tag']}' already exists. Please provide a different tag: ")
            if new_tag.strip() == "":
                print("Invalid tag. Skipping log save.")
                return
            self.logs['trace_tag'] = new_tag.strip()
            self.save_logs_to_file(directory)
            return
        
        with open(filename, 'w') as file:
            json.dump(self.logs, file, indent=4)


