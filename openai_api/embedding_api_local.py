'''
Author: AnchoretY
Date: 2023-08-24 23:19:58
LastEditors: AnchoretY
LastEditTime: 2023-08-25 04:01:53
'''

import os 
os.environ['OPENAI_API_BASE'] = "http://localhost:8000/v1"
os.environ['OPENAI_API_KEY'] = "EMPTY"

model_id = "text-embedding-ada-002"

from langchain.embeddings import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(model=model_id) # langchain的 embedding 模型必须采用 openai 的embedding 模型名称
embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)
print(embeddings[0])
print(len(embeddings), len(embeddings[0]))

