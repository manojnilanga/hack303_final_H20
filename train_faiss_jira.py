from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from config import *
import time



all_docs = []
# csvs = ["./train/PNL_ADL_JIRACloud_1.csv", "./train/PNL_ADL_JIRAServer_1.csv", "./train/PNL_ADL_JIRAServer_2.csv", "./train/commit_details.csv"]
csvs = ["./train/PNL_ADL_JIRACloud_1_chunk.csv", "./train/commit_details_chunk.csv"]
for csv in csvs:
    loader = CSVLoader(csv)
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
vector.save_local("faiss_jira_chunk_v2")


# # for all data
#
# document_count_per_set = 2000
# amount_of_document_sets = len(documents)//document_count_per_set + 1
#
# for i in range(0,amount_of_document_sets):
#     print("remaining datasets: " + str(amount_of_document_sets-i))
#     start_doc = i*document_count_per_set
#     end_doc = (i+1)*document_count_per_set
#     if(end_doc>len(documents)):
#         end_doc = len(documents)
#     print("start_doc: " + str(start_doc))
#     print("end_doc: " + str(end_doc))
#     doc = documents[start_doc:end_doc]
#     while(True):
#         try:
#             print("creating vectors ...")
#             if(i==0):
#                 vector = FAISS.from_documents(doc, embeddings)
#             else:
#                 new_vector = FAISS.from_documents(doc, embeddings)
#                 vector.merge_from(new_vector)
#             break
#         except Exception as e:
#             print(str(e))
#             print("sleeping .. 70 second")
#             time.sleep(70)
#     vector.save_local("faiss_jira_chunk")
# print("end")


