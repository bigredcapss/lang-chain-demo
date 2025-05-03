# -*- coding: UTF-8 -*-

# filename : test_cosplay_chat_bot.py
# description : 基于langchain构建一个基于角色扮演的聊天机器人
# author by : peanut
# date : 2025/5/1


"""
    实现一个基于角色扮演的聊天机器人
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


chat_model = ChatOpenAI(model="gpt-4o-mini")

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你现在扮演孔子的角色，尽量按照孔子的风格回复，不要出现‘子曰’",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

with_message_history = RunnableWithMessageHistory(
    prompt | chat_model,
    get_session_history
)

config = {"configurable": {"session_id": "dreamhead"}}


while True:
    user_input = input("You:> ")
    if user_input.lower() == 'exit':
        break
    stream = with_message_history.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    for chunk in stream:
        print(chunk.content, end='', flush=True)
    print()