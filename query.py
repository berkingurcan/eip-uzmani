import glob
import os
import openai
import tiktoken
import pinecone
import json
import time
import requests
import numpy as np

from uuid import uuid4
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

def retrieval(query):
    # TODO retrival with query

    openai.api_key = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'
    # Load Pinecone API key
    api_key = os.getenv('PINECONE_API_KEY') or 'YOUR_API_KEY'
    # Set Pinecone environment. Find next to API key in console
    env = os.getenv('PINECONE_ENVIRONMENT') or "YOUR_ENV"

    embed_model = "text-embedding-ada-002"

    chat = ChatOpenAI(openai_api_key=openai.api_key)

    embed = OpenAIEmbeddings(
        model=embed_model,
        openai_api_key=openai.api_key
    )

    pinecone.init(api_key=api_key, environment=env)
    index = pinecone.Index('mango')
    vector_store = Pinecone(index, embed.embed_query, "text")

    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    xq = openai.Embedding.create(input=query, engine=embed_model)['data'][0]['embedding']
    res = index.query([xq], top_k = 3, include_values=True, include_metadata=True)
    print(res)

    return qa.run(query)

result = retrieval("I want to create erc20 token contract on OP Stack how to do that")
print(result)

def print_index_stats():
    index = pinecone.Index('mango')
    print(index.describe_index_stats())

print_index_stats()