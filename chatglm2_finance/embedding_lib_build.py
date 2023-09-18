'''
Author: AnchoretY
Date: 2023-08-15 03:56:44
LastEditors: AnchoretY
LastEditTime: 2023-09-05 04:38:56
'''

# 基本方案：
#    1. 根据提问确定公司名称与年报年份
#    2. 找到对应年报文件进行关联向量查找


# 面临问题：
    # 1. 模型允许输入的最大长度问题
    # 2. 在参考文献中没有相关内容时，GPT3.5 可以很好的回答参考文献中没有相关信息，但 vicuna 还是会幻觉->不在文件段落中添加文件名称
    # 3. 文件名为公司名，但是提问有可能为文件内容
    # 4. vicuna、gpt等开源模型并不会在输出内容时严格区分词性，如让它总结利好信息，他会把一些不利信息也总结为利好信息。更加倾向于做概括，这应该是模型训练中大多都是总结性问题的缘故，但是GPT、Claude并没有这个问题。   这里可能和RLHF相关？
    # 5. 生成莫须有的信息，参考文献中并不存在，但是输出内容中存在。例如让其总结造纸业利好消息，模型捏造金龙鱼、维达等信息，参考文献中并没有相关内容


#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
"""
import os
import json
import pandas as pd
from tqdm import tqdm

from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter,TextSplitter,RecursiveCharacterTextSplitter

question = "森海林业在 2020年总收入是多少?"

# 获取对应指定文件列表
root_path = "/home/yhk/github/DeepSpeed/llm_dataset/modelscope/chatglm_llm_fintech_raw_dataset/alltxt/"
file_l = []
for file in os.listdir(root_path):
    if "岳阳林纸" in file and "2020" in file:
        file_l.append(file)

print("找到相关文档:")
print("\t\n".join(file_l))

# 目标文档向量化处理
# 数据类型：{'text', '页脚', '页眉', 'excel'}
docs = []
idx = 0 
for filename in tqdm(file_l):
    file_content = ""
    file = os.path.join(root_path,filename)
    with open(file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            if data=={}:
                file_content+="\n"
                continue
            if data['type']=='页眉' or data['type']=='页脚':
                continue
            elif data['type']=='excel':
                continue
            else:
            # print(data['type'],data['inside'])
                file_content+=data['inside']

        docs.append(Document(page_content=file_content,metadata={'filename':filename}))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=20,length_function=len)
docs = text_splitter.split_documents(docs)

# embedding 模型加载，多种模型选择对比
# embedding_model_name = "../DeepSpeed/llm_model/WangZeJun/simbert-base-chinese"
# embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
from langchain.embeddings import OpenAIEmbeddings
os.environ['OPENAI_API_BASE'] = "http://localhost:8000/v1"
os.environ['OPENAI_API_KEY'] = "EMPTY"
embedding_model = OpenAIEmbeddings()

db = FAISS.from_documents(docs, embedding_model)
db.save_local('cache/lol/')

# 查找最相关 k 个文档片段
relate_docs = db.similarity_search(question,k=3)
print(" 相关文档片段：")
print(relate_docs)


# 调用模型对相关文档进行总结
from langchain.prompts import PromptTemplate
t1 = """【提问】
{question}
【任务要求】
请按照如下要求回答提问：
1. 去读并理解背景知识
2. 如果参考信息中没有提供足够的信息，那么直接回复:未提及
3. 在回答未提及前请仔细检查是否存在相关内容
4. 忽略背景知识中与问题无关的信息
5. 回答以”答："开始
6. 尽可能分点回答
【背景知识】
{relate_doc1}
{relate_doc2}
"""
prompt_template = PromptTemplate(
    template=t1,
    input_variables=["question","relate_doc1","relate_doc2"]
)


prompt = prompt_template.format(
    question=question,
    relate_doc1=relate_docs[0].page_content,
    relate_doc2=relate_docs[2].page_content,
)


import openai
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"
model_id = "vicuna-13b-v1.5"
# model_id = "chatglm2-6b"


# openai.api_type = "azure"
# openai.api_version= "2023-05-15"
# openai.api_base = "https://chatso-gpt-50.openai.azure.com/"
# openai.api_key = "4445ee730d354b95af26309da37d34f4"
# model_id = "gpt-35-turbo"

completion = openai.ChatCompletion.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                frequency_penalty=2
            )
            
# completion = openai.ChatCompletion.create(
#                     engine=model_id,
#                     messages=[{"role": "user", "content": prompt}]
#                 )
ret = completion.choices[0].message.content
print(prompt)

print(ret)




