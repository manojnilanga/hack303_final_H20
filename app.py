import streamlit as st
from openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from config import *
import re

st.title("Code Review Assistant")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

vector = FAISS.load_local("faiss_jira_chunk_v2", embeddings, allow_dangerous_deserialization=True)
retriever = vector.as_retriever()

vector_pdf = FAISS.load_local("faiss_pdf_chunk", embeddings, allow_dangerous_deserialization=True)
retriever_pdf = vector_pdf.as_retriever()

model = ChatOpenAI(openai_api_key=OPENAI_KEY)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt_temp = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
chain = setup_and_retrieval | prompt_temp | model | output_parser

setup_and_retrieval_pdf = RunnableParallel(
    {"context": retriever_pdf, "question": RunnablePassthrough()}
)
chain_pdf = setup_and_retrieval_pdf | prompt_temp | model | output_parser

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    if("page" in prompt.lower() or "section" in prompt.lower()):
        matches = re.findall(JIRA_REGEX, prompt)
        if(len(matches)>0):
            jira_id = matches[0]
            response_jira_message = chain.invoke("what is the description for " + jira_id)
            print(response_jira_message)
            response_pdf_message = chain_pdf.invoke("Give the relevant sections to refer for this description: " + response_jira_message)
            print(response_pdf_message)
            response = response_jira_message + "\n\n" + response_pdf_message
        else:
            response = chain_pdf.invoke(prompt)
            print(response)
    else:
        response = chain.invoke(prompt)
        print(response)
    print("------------------")
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})