'''
Author: AnchoretY
Date: 2023-08-15 06:11:06
LastEditors: AnchoretY
LastEditTime: 2023-08-28 04:01:02
'''
# 作用：根据提问抽取上市公司名称与年份
# 问题：1. 开源模型很难严格按照 json 格式进行数据提取，few-shot 依旧不能解决


import json
import openai
import argparse
from tqdm import tqdm
from langchain import PromptTemplate

t1 = """从下面的问题中按照 json 格式提取询问公司的名称与年份,未出现公司或年份使用填入空,注意不要输出任何其他内容，下面是几个提取示例:\n
###
指令:请将下面的问题中提到的公司名和年份按照 json 格式进行提取，不要输出任何其他信息
问题:请根据江化微2019年的年报，简要介绍报告期内公司主要销售客户的客户集中度情况，并结合同行业情况进行分析。
{{"company":["江化微"],"year":[2019]}}

指令:请将下面的问题中提到的公司名和年份按照 json 格式进行提取，不要输出任何其他信息
问题:2023年四方科技电子信箱是什么?
{{"company":["四方科技"],"year":[2023]}}

指令:请将下面的问题中提到的公司名和年份按照 json 格式进行提取，不要输出任何其他信息
问题:研发费用对公司的技术创新和竞争优势有何影响？
{{"company":[],"year":[]}}

指令:请将下面的问题中提到的公司名和年份按照 json 格式进行提取，不要输出任何其他信息
问题:南京康尼机电股份有限公司2019年企业研发经费与利润比值是多少?保留2位小数。
{{"company":["南京康尼机电股份有限公司"], "year":[2019]}}

指令:请将下面的问题中提到的公司名和年份按照 json 格式进行提取，不要输出任何其他信息
问题:您所在公司财务报告中提到了2021年第一季度的营收增长情况，请问这个增长是以幅度或速度来衡量的？
{{"company":[],"year":[2021]}}
### 

指令:请将下面的问题中提到的公司名和年份按照 json 格式进行提取，不要输出任何其他信息
问题:{question}
"""

def entity_extract(question,model_id):
    if model_id=='gpt-35-turbo':
        openai.api_type = "azure"
        openai.api_version= "2023-05-15"
        openai.api_base = "https://chatso-gpt-50.openai.azure.com/"
        openai.api_key = "4445ee730d354b95af26309da37d34f4"
        model_id = "gpt-35-turbo"
    else:
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"
        model_id = model_id

    prompt_template = PromptTemplate(
    template=t1,
    input_variables=["question"]
    )

    prompt = prompt_template.format(question=question)

    if model_id=='gpt-35-turbo':
        completion = openai.ChatCompletion.create(
            engine=model_id,
            messages=[{"role": "user", "content": prompt}]
        )
    else:
        completion = openai.ChatCompletion.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            frequency_penalty=2
        )
    ret = completion.choices[0].message.content
    return ret



if __name__=='__main__':
    model_id = "vicuna-13b-v1.5"
    with open("test_questions.jsonl") as f:
        with open("entity.jsonl",'w') as fw:
            for i,line in tqdm(enumerate(f.readlines())):
                data = json.loads(line)
                ret = entity_extract(data['question'],model_id)
                flag = False
                while flag==False:
                    try:
                        ret = json.loads(ret)
                        json.dump(ret,fw)
                        fw.write("\n")
                        flag = True
                    except:
                        pass
                if i==10:
                    break

            
        



