from dataloader.loader import loadDataset
from rag.search import RAG

print("Starting main ....")


def main():

    # take user input of path of documents
    file_path = ""

    # load documents
    data = loadDataset(file_path)

    # take user input of prompt
    query = ""

    # run RAG
    response = RAG(query, data)
    print(response)


if __name__ == "__main__":

    main()
