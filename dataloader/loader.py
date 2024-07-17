# from PyPDF2 import PdfReader
from pathlib import Path
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import glob
import json 

from .chunking import chunkFileByChunkSize

# Load environment variables from .env file
load_dotenv()

client = OpenAI()

def embedDocument(file_path: Path):
    file_path = Path(file_path)
    # check if already embedded 
    embedded_file_path = file_path.parent / f"Embedded_{file_path.stem}.csv"
    if embedded_file_path.exists():
        #print("Embeddings table for doc already exists. Loading it!")
        # load embeddings table into df
        directory = file_path.parent
        filename = file_path.name
    
        # Load the CSV file with semicolon separator
        df = pd.read_csv(embedded_file_path, sep=';')
        
        # Convert JSON strings back to lists
        df['Embeddings'] = df['Embeddings'].apply(json.loads)
        return df
    
    # if not yet embedded, embed
    else:     
        print("Embeddings table doesnt exist. Embedding documents...") 
        
        # generate embeddings table 
        list_chunks,list_pagenum = chunkFileByChunkSize(file_path)
        filename = file_path.name
        
        df = pd.DataFrame({
            'TextChunk': list_chunks,
            'PageNum': list_pagenum,
            'Filename': filename,
        })
        
        # generate embeddings on the TextChunk column
        df['Embeddings'] = df.TextChunk.apply(lambda x: generateEmbedding(x, model='text-embedding-3-small'))
        df['ChunkLength'] = df.TextChunk.apply(lambda x: len(x))
        df = df[["Filename", "TextChunk","ChunkLength","PageNum","Embeddings"]]
        
        # saving 
        directory = file_path.parent
        filename = file_path.name
        
        # Construct the embedded file path
        embedded_filename = 'Embedded_' + filename.replace('.pdf', '.csv')
        embedded_file_path = directory / embedded_filename
        
        df.to_csv(embedded_file_path, index=False, sep=';')
        print("saved embeddings file")
        return df

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def loadDataset(file_path: Path) -> pd.DataFrame:
    list_chunks, list_pagenum = chunkFileByChunkSize(file_path)

    df = pd.DataFrame({
        'TextChunk': list_chunks,
        'PageNum': list_pagenum
    })

    df['Embeddings'] = df.TextChunk.apply(
        lambda x: generateEmbedding(x, model='text-embedding-3-small'))
    df['ChunkLength'] = df.TextChunk.apply(lambda x: len(x))
    df = df[["TextChunk", "ChunkLength", "PageNum", "Embeddings"]]

    return df


def loadEmbeddings(folder_path: Path):
    pdf_files = list(folder_path.glob('*.pdf'))

    columns = ['Filename', 'TextChunk', 'ChunkLength', 'PageNum', 'Embeddings']
    df_final = pd.DataFrame(columns=columns)

    for file in pdf_files:
        df = embedDocument(file)
        df_final = pd.concat([df_final, df], ignore_index=True)
    return df_final