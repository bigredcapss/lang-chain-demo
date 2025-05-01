# -*- coding: UTF-8 -*-

# filename : test_prompt_template.py
# description : LangChain的提示词模版PromptTemplate抽象-->具体实现为ChatPromptTemplate
# author by : peanut
# date : 2025/5/1

"""
    1、LangChain 的 PromptTemplate 抽象，它负责将用户输入的 Prompt 模板转换为 LangChain 的 PromptValue 对象，
    这个对象可以被 ChatModel 接受，并最终转换为具体的模型输入。
    2、引入了 PromptTemplate 之后，开发者写的提示词和用户的消息就完全分开了。开发者可以不断地调整提示词以便达到更好的效果，
    而这一切对用户完全屏蔽掉了。此外，还有一个好处，就是好的提示词模板是可以共享出来的，我们甚至可以把别人写好的提示词用在自己的代码里。所以提示词社区就出来了。

"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os

# 从环境变量获取配置
os.environ["OPENAI_API_KEY"] = "your api key" 
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Translate the following from English into Chinese:"),
        ("user", "{text}")
    ]
)

model = ChatOpenAI(model="gpt-4o-mini")

# 通过 LCEL 把 prompt_template 和 model 组成一条链
chain = prompt_template | model

# 触发链式调用
result = chain.invoke({"text":"Welcome to LLM application development!"})

print(result)
