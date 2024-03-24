from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.json_loader import JSONLoader
from config import *

all_docs = []

jsons = ["./train/pull_request_data_chunk.json"]
for json in jsons:
    loader = JSONLoader(json, jq_schema='.', text_content=False)
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
vector.save_local("faiss_pull_chunk")

