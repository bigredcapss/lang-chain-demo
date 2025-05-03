# -*- coding: UTF-8 -*-

# filename : test_history_chat_bot.py
# description : 基于langchain构建一个可以记事的聊天机器人
# author by : peanut
# date : 2025/5/1


"""
为了支持聊天历史，LangChain 引入了一个抽象叫 ChatMessageHistory。
为了简单，我们这里使用了 InMemoryChatMessageHistory，也就是在内存存放的聊天历史。
有了聊天历史，就需要把聊天历史和模型结合起来，这就是 RunnableWithMessageHistory 所起的作用，
它就是一个把聊天历史和链封装到一起的一个类。

这里的 Runnable 是一个接口，它表示一个工作单元。我们前面说过，组成链是由一个一个的组件组成的。
严格地说，这些组件都实现了 Runnable 接口，甚至链本身也实现了 Runnable 接口，
我们之前讨论的 invoke、stream 等接口都是定义在 Runnable 里，可以说，Runnable 是真正的基础类型，
LCEL 之所以能够以声明式的方式起作用，Runnable 接口是关键。
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory



chat_model = ChatOpenAI(model="gpt-4o-mini")


# 创建一个内存中的聊天历史记录存储
store = {}

# 定义一个函数，用于根据会话ID获取聊天历史记录
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 创建一个带有聊天历史的模型
with_message_history = RunnableWithMessageHistory(chat_model, get_session_history)

# 配置会话ID
config = {"configurable": {"session_id": "dreamhead"}}

while True:
    user_input = input("You:> ")
    if user_input.lower() == 'exit':
        break
    stream = with_message_history.stream(
        [HumanMessage(content=user_input)],
        config=config
    )
    for chunk in stream:
        print(chunk.content, end='', flush=True)
    print()