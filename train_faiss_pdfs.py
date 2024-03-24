from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from config import *

all_docs = []

import os

def get_file_names(folder_path):
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)
    return file_names

folder_path = "./train/pdfs"
pdfs = get_file_names(folder_path)
print("Files in the folder:")
for file_name in pdfs:
    print(file_name)

for pdf in pdfs:
    loader = PyPDFLoader("./train/pdfs/"+pdf)
    docs = loader.load()
    all_docs.extend(docs)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=20,
                                               length_function=len,
                                               is_separator_regex=False, )

documents = text_splitter.split_documents(all_docs)
print("total documents count: " + str(len(documents)))
vector = FAISS.from_documents(documents, embeddings)
vector.save_local("faiss_pdf_chunk")

