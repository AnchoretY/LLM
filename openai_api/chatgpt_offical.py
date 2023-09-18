'''
Author: AnchoretY
Date: 2023-09-18 05:22:29
LastEditors: AnchoretY
LastEditTime: 2023-09-18 05:24:55
'''
import os
from langchain.schema import HumanMessage
from langchain.chat_models import AzureChatOpenAI


class ChatGPT():
    def __init__(self,temperature=0,max_tokens=4000):
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_VERSION"] = "2023-05-15"
        os.environ["OPENAI_API_BASE"] = "https://chatso-gpt-50.openai.azure.com/"
        os.environ['OPENAI_API_KEY']="xxx"

        self.llm = AzureChatOpenAI(
            deployment_name="gpt-35-turbo",
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def get_response(self,prompt):
        return self.llm([HumanMessage(content=prompt)]).content


if __name__=="__main__":
    prompt = "天气怎么样?"
    llm = ChatGPT()
    response = llm.get_response(prompt)
    print(response)