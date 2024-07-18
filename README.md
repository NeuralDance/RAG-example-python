# RAG Example with Logging

![Example Image](RAG_overview.png)

This is an example for a very basic RAG architecture using **semantic search (kNN)**, **keyword search (BM25)** and **reranking with RFF**.

We dont use any vector database, but loads all embeddings into memory. Hence this example will not work for large amounts of data. The purpose of this repo is to help RAG n00bs to familiarize with basic RAG architectures by building a RAG themselves without relying on off-the-shelf frameworks like langchain etc.

## Deatils to the RAG Implementation

### Pre-Set-Up: Loading Data from PDF files

If you want to have some test data download the PDFs [here](https://drive.google.com/drive/folders/1PWICaG6HF5EtmmN23fs8-UZDxHz05Y_y?usp=sharing). Save the docs in a folder and add the folder path in your `.env` file under `FOLDER_PATH_DOCUMENTS` (there you also need to set your OpenAI key!).

The code takes the folder, checks for all PDFs and embeds the ones that are not embedded (Embedded files are saved in a csv file in the same folder). We chunk each page of a PDF into multiple embeddings. Each embedding is max. 800 characters long while we try to not cut off in the middle of words/sentences. We clean the text chunks for special characters. All the chunks are embedded using OpenAI's `text-embedding-3-small`.

### 1) User Query

The incoming user query is embedded using OpenAI's `text-embedding-3-small`, similar to all text chunks in our document collection. The query is then routed to both the Keyword Search and the Semantic Search.

### 2a) Semantic Search - kNN

Using k-Nearest Neighbours and the cosine similarity, we find the closest k embeddings in the document collection to the embedding of the user query.

<!-- Why do we use both? Are we evaluating two measures of similarity to then compare or combine them? -->

### 2b) Keyword Search - BM25

We use BM25 for keyword search - BM for best match.

More info on BM25: https://en.wikipedia.org/wiki/Okapi_BM25

### 3) Reranking - Reciprocal Rank Fusion (RFF)

We now have two rankings of all text chunks in our document collection: A ranking by keyword search and a ranking by semantic search. To combine these into one ranking we use Reciprocal Rank Fusion (RFF). A more advanced (but paid) alternative woule be Cohere's Reranker.

### 4) Output Generation - LLM Request

We generate the final answer with an LLM. We give the LLM a very basic system prompt and a text string consisting of the top k embeddings, ranked by the Reciprocal Rank Fusion (RFF).

Working with RAG you will soon realise it still is an immature technoloy. We might add more compley steps to this RAG for presentation purposes.

## How to run this?

In the root of the repo run.

```
python main.py
```

You will be asked to provide your query. If you run this the first time it might tske a bit as the documents need to be embedded (same if you add more documents later).

Happy testing!

## How to use Observability?

During each run, we automatically log the RAG data to provide maximum observability and make both debugging and production easier. You can disable this setting `logging = False` in `main.py`. Your logs are saved under `rag/observe/logs` in `.json` files. You can set a `trace_tag` in the `main.py` function when running a RAG search to easier find your belonging logs. Examples how to visualise the logs (in more readable format than JSON) use the `Agent-Observability-Visualizer.ipynb` notebook.

### What is logged?

- **Trace Metadata**: Trace tag, trace ID, timestamp, git commit
- **User Query**
- **Keyword Search**: We log the top ranked text chunks with its keyword search score.
- **Semantic Search**: We log the top ranked text chunks with its semantic search score.
- **Reranked Results**: We log the reranked results. A table of the top ranked text chunks after reranking, providing the RRF, semantic and keyword search score per embedding.
- **LLM Prompt**: The LLM prompt provided to the LLM to generate the final response. This includes the system prompt, the user query and the text chunks retrieved by the RAG.
- **Final LLM Response**
