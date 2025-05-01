# -*- coding: UTF-8 -*-

# filename : test_chat_model.py
# description : LangChain的ChatModel抽象-->具体实现为ChatOpenAI
# author by : peanut
# date : 2025/5/1


"""
    1、既然 LangChain 是为了构建大模型应用而生的，其最核心的基础抽象一定就是聊天模型（ChatModel）;

    2、我们知道，LangChain提供抽象，具体的实现是由社区生态提供的，所以，我们要想使用 ChatOpenAI 模型，需要安装 langchain-openai 这个包。
    有许多服务提供商会提供多个基础抽象的实现，比如，OpenAI 除了 ChatModel 之外，还提供了 Embedding 模型，
    所有与 OpenAI 相关的内容都会统一放到 langchain-openai这个包里，LangChain 社区将它们统一称为供应商（Provider），这里的 OpenAI 就是一个供应商。

    3、LangChain 应用代码核心就是构建一条链。在这里，单独的一个模型也是一条链，只不过这条链上只有一个组件，它就是 ChatModel；

"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os

# 从环境变量获取配置
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"  # 替换为您的 API 密钥
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"

messages = [
    SystemMessage(content="Translate the following from English into Chinese:"),
    HumanMessage(content="Welcome to LLM application development!"),
]

model = ChatOpenAI(model="gpt-4o-mini")

# 普通应答
result = model.invoke(messages)
print(result)


# 流式应答
stream = model.stream(messages)
for response in stream:
    print(response.content, end="|")