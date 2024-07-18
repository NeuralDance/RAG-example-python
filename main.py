from pathlib import Path
from dataloader.loader import loadDataset,loadEmbeddings
from rag.search import RAG
from dotenv import load_dotenv
import os
from examplary_prompts import EXAMPLE_PROMPTS
from rag.helper import get_random_element
load_dotenv()

def main(user_input = True,logging = True, trace_tag=None):
    # take user input of path of documents
    folder_path = Path(os.getenv('FOLDER_PATH_DOCUMENTS'))

    # load documents
    data = loadEmbeddings(folder_path)

    # take user input of prompt
    if user_input:
        query = input("Whats your question? ")
    else:
        query = get_random_element(EXAMPLE_PROMPTS)

    # run RAG
    response = RAG(query, data,logging,trace_tag)

    # Display the results in the terminal 
    print(query,"\n\n")
    print(response)


if __name__ == "__main__":
    main(user_input = False,trace_tag="Test1")
