'''
Author: AnchoretY
Date: 2023-08-24 23:19:58
LastEditors: AnchoretY
LastEditTime: 2023-08-25 04:08:11
'''

import openai
import os 
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"
# os.environ['OPENAI_API_BASE'] = "http://localhost:8000/v1"
# os.environ['OPENAI_API_KEY'] = "EMPTY"

prompt = """【提问】
2021年濑粉利润增长率为多少？
【任务要求】
请按照如下要求回答提问：
1. 去读并理解背景知识
2. 如果参考信息中没有提供足够的信息，那么直接回复:根据已知信息无法计算
【背景知识】
2019年濑粉公司利润200亿元
2020年濑粉公司营收230亿元，利润为210亿元
"""

model_id = "vicuna-13b-v1.5"
# model_id = "chatglm2-6b"

completion = openai.ChatCompletion.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                frequency_penalty=2
            )

ret = completion.choices[0].message.content
print(prompt)
print(ret)