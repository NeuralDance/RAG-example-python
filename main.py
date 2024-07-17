from dataloader.loader import loadDataset,loadEmbeddings
from rag.search import RAG
from dotenv import load_dotenv
import os
load_dotenv()

def main():
    # load PDFs from specified folder
    data = loadEmbeddings(os.getenv('FOLDER_PATH_DOCUMENTS'))

    # take user input of prompt
    # query = "Can you give me information about the description of the warning symbols?"
    query = input("Whats your question? ")

    # run RAG
    response = RAG(query, data)

    # Display the results in the terminal 
    print(query,"\n\n")
    print(response)


if __name__ == "__main__":
    main()
