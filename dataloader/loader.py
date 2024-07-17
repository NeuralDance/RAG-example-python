# from PyPDF2 import PdfReader
import re
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import glob
import json 

from .chunking import split_text_into_chunks, chunkFileByChunkSize

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

import os

def check_embedded_file_existence(file_path):
    # Extract directory and filename from the original file path
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)

    # Construct the embedded file path
    embedded_filename = 'Embedded_' + filename.replace('.pdf', '.csv')
    embedded_file_path = os.path.join(directory, embedded_filename)

    # Check if the embedded file exists
    if os.path.exists(embedded_file_path):
        #print(f"The embedded file {embedded_filename} exists in the directory.")
        return True
    else:
        #print(f"The embedded file {embedded_filename} does not exist in the directory.")
        return False


def embedDocument(file_path):

    # check if already embedded 
    if check_embedded_file_existence(file_path):
        #print("Embeddings table for doc already exists. Loading it!")
        # load embeddings table into df
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
    
        # Construct the embedded file path
        embedded_filename = 'Embedded_' + filename.replace('.pdf', '.csv')
        embedded_file_path = os.path.join(directory, embedded_filename)
    
        # Load the CSV file with semicolon separator
        df = pd.read_csv(embedded_file_path, sep=';')
        
        # Convert JSON strings back to lists
        df['Embeddings'] = df['Embeddings'].apply(json.loads)
        return df
    
    else:
        
        print("Embeddings table doesnt exist. Embedding documents...") 
        # generate embeddings table 
        list_chunks,list_pagenum = chunkFileByChunkSize(file_path)
        filename = os.path.basename(file_path)
        
        df = pd.DataFrame({
            'TextChunk': list_chunks,
            'PageNum': list_pagenum,
            'Filename': filename,
        })
        
        df['Embeddings'] = df.TextChunk.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
        df['ChunkLength'] = df.TextChunk.apply(lambda x: len(x))
        df = df[["Filename", "TextChunk","ChunkLength","PageNum","Embeddings"]]
        
        # saving 
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        # Construct the embedded file path
        embedded_filename = 'Embedded_' + filename.replace('.pdf', '.csv')
        embedded_file_path = os.path.join(directory, embedded_filename)
        
        df.to_csv(embedded_file_path, index=False, sep=';')
        print("saved embeddings file")
        return df


def getAllPdfInFolder(folder_path):   
    # Ensure the folder path ends with a slash for correct file joining
    if not folder_path.endswith('/'):
        folder_path += '/'
    
    # Use glob to find all PDF files in the folder
    pdf_files = glob.glob(folder_path + '*.pdf')
    
    return pdf_files

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
    df['ChunkLength'] = df.TextChunk.apply(lambda x: len(x))
    df = df[["TextChunk", "ChunkLength", "PageNum", "Embeddings"]]

    return df


def loadEmbeddings(folder_path):
    pdf_files = getAllPdfInFolder(folder_path)

    columns = ['Filename', 'TextChunk', 'ChunkLength', 'PageNum', 'Embeddings']
    df_final = pd.DataFrame(columns=columns)

    for file in pdf_files:
        df = embedDocument(file)
        df_final = pd.concat([df_final, df], ignore_index=True)
    return df_final