from PyPDF2 import PdfReader
from pathlib import Path
import re
import pandas as pd
import typing as t


def split_text_into_chunks(text: str, n: int = 800) -> t.List[str]:
    """
    chunks pdfs by chunksize of n.
    
    
    :params text: text of pdf
    :params n: chunksize
    :returns: list of chunks
    """

    if n <= 0:
        raise ValueError("Chunk size (n) must be greater than 0")

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= n:
            # Add the sentence to the current chunk
            current_chunk += sentence + " "
        else:
            if current_chunk:
                # If the current chunk is not empty, add it to the list
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Split the sentence into smaller parts without cutting words
            words = sentence.split()
            for word in words:
                if len(current_chunk) + len(word) + 1 <= n:  # +1 for space
                    current_chunk += word + " "
                elif not current_chunk:
                    # If no words can fit, break the word itself
                    while word:
                        chunks.append(word[:n].strip())
                        word = word[n:]
                else:
                    # Add the current chunk to the list and start a new one
                    chunks.append(current_chunk.strip())
                    current_chunk = word + " "

    # Add any remaining content to the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def chunkFileByChunkSize(file_path: Path) -> t.Tuple[t.List[str], t.List[int]]:
    """
    function reads file either with PDFReader for pdfs or with UnstructuredExcel Reader for excel files and 
    splits content chunks

    :param file_path: path to file in bucket

    :returns: list of chunks and pagenumbers

    """

    if file_path.suffix == '.pdf':

        reader = PdfReader(file_path)

        list_chunks: t.List[str] = []
        list_pagenum: t.List[int] = []

        for i in range(0, len(reader.pages)):
            pagenum = i + 1
            text_page = reader.pages[i].extract_text()

            replacements = {
                "..": "",
                "\n": "",
                "•": "",
                "  ": "",
                "\uf071": "-",
                "\uf06e": "-",
                "\uf0a6": "-",
                "\uf0d8": "-"
            }

            for old, new in replacements.items():
                text_page = text_page.replace(old, new)

            chunks = split_text_into_chunks(text_page)

            for c in chunks:
                if len(c) > 0:
                    list_chunks.append(c)
                    list_pagenum.append(pagenum)

        if len(list_pagenum) != len(list_chunks):
            raise ValueError(
                "Length of page numbers and text chunks doesn't match!")

        return list_chunks, list_pagenum    
    else:
        return [], []
