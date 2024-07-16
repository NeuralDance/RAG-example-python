# from PyPDF2 import PdfReader
import re
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

from .chunking import split_text_into_chunks, chunkFileByChunkSize

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def loadDataset(file_path):
    list_chunks, list_pagenum = chunkFileByChunkSize(file_path)

    df = pd.DataFrame({
        'TextChunk': list_chunks,
        'PageNum': list_pagenum
    })

    df['Embeddings'] = df.TextChunk.apply(
        lambda x: get_embedding(x, model='text-embedding-3-small'))
    df['ChunkLegth'] = df.TextChunk.apply(lambda x: len(x))
    df = df[["TextChunk", "ChunkLegth", "PageNum", "Embeddings"]]

    return df
