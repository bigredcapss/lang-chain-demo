# -*- coding: UTF-8 -*-

# filename : test_chat_bot.py
# description : 基于langchain构建一个不会记事的聊天机器人
# author by : peanut
# date : 2025/5/1


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

chat_model = ChatOpenAI(model="gpt-4o-mini")


# 无法记事的聊天机器人
while True:
    user_input = input("You:> ")
    if user_input.lower() == 'exit':
        break
    stream = chat_model.stream([HumanMessage(content=user_input)])
    for chunk in stream:
        print(chunk.content, end='', flush=True)
    print()

