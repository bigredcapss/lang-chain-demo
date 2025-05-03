# -*- coding: UTF-8 -*-

# filename : test_trim_messages_cosplay_chat_bot.py
# description : 基于langchain构建一个控制消息规模的角色扮演聊天机器人
# author by : peanut
# date : 2025/5/1


"""
    在实现聊天机器人的过程中，还有一个很现实的问题，我们需要处理一下。如果不加任何限制，所有的聊天历史都会附加到新的会话中，
    随着聊天的进行，聊天历史很快就会超过大模型的上下文窗口大小。一种典型处理办法是，对聊天历史进行限制。

    LangChain 提供了一个 trim_messages 用来控制消息的规模，它提供了很多控制消息规模的参数：
        1、max_tokens，限制最大的 Token 数量。
        2、strategy，限制的策略，从前面保留（first），还是从后面保留（last）。
        3、allow_partial，是否允许把拆分消息。
        4、include_system，是否要保留最开始的系统提示词。
    这其中最关键的是就是 max_tokens，这也是我们限制消息规模的主要原因。不过，这里是按照 Token 进行计算，
    我们该怎么计算 Token 呢？对 OpenAI API 来说，一种常见的解决方案是采用 tiktoken，这是一个专门用来处理的 Token 的程序库。
"""



from typing import List
import tiktoken
from langchain_core.messages import SystemMessage, trim_messages, BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# 定义一个函数，用于计算字符串的 Token 数量
def str_token_counter(text: str) -> int:
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))

# 定义一个函数，用于计算消息的 Token 数量
def tiktoken_counter(messages: List[BaseMessage]) -> int:
    num_tokens = 3
    tokens_per_message = 3
    tokens_per_name = 1
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported messages type {msg.__class__}")
        num_tokens += (
                tokens_per_message
                + str_token_counter(role)
                + str_token_counter(msg.content)
        )
        if msg.name:
            num_tokens += tokens_per_name + str_token_counter(msg.name)
    return num_tokens


chat_model = ChatOpenAI(model="gpt-4o-mini")


# 创建一个内存中的聊天历史记录存储
store = {}

# 定义一个函数，用于根据会话ID获取聊天历史记录
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

# 创建一个带有聊天历史的模型
trimmer = trim_messages( 
    max_tokens=4096, 
    strategy="last", 
    token_counter=tiktoken_counter, 
    include_system=True,)

# 创建一个带有聊天历史的模型
with_message_history = RunnableWithMessageHistory( trimmer | prompt | chat_model, get_session_history)


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



