# -*- coding: UTF-8 -*-

# filename : test_rag_chat_bot.py
# description : 基于Chroma实现RAG聊天机器人
# author by : peanut
# date : 2025/5/4

"""
    在 LangChain 代码里， | 运算符被用作不同组件之间的连接，其实现的关键就是大部分组件都实现了 Runnable 接口，
    在这个接口里实现了 __or__ 和 __ror__。__or__ 表示这个对象出现在| 左边时的处理，相应的 __ror__ 表示这个对象出现在右边时的处理

"""

from operator import itemgetter
from typing import List
import tiktoken
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage, trim_messages
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_chroma import Chroma

# 指定要检索的向量数据库
vectorstore = Chroma(
    collection_name="ai_learning",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="vectordb"
)

# 创建检索器，使用相似性检索
"""
    Retriever。它就是充当 RAG 中的 R。Retriever 的核心能力就是根据文本查询出对应的文档（Document）。
    Retriever 并不只有向量数据库一种实现，比如，WikipediaRetriever 可以从 Wikipedia 上进行搜索。
    所以，一个 Retriever 接口就把具体的实现隔离开来。

    回到向量数据库上，当我们调用 as_retriever 创建 Retriever 时，还传入了搜索类型（search_type），
    这里的搜索类型和前面讲到向量数据库的检索方式是一致的，这里我们传入的是 similarity，当然也可以传入 mmr。
    文档检索出来，并不能直接就和我们的问题拼装到一起。这时，就轮到提示词登场了。
    下面是我们在代码里用到的提示词（改造自https://smith.langchain.com/hub/rlm/rag-prompt）

"""
retriever = vectorstore.as_retriever(search_type="similarity")

# 计算字符串的token数量
def str_token_counter(text: str) -> int:
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))

# 计算消息的token数量
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

# 修剪消息，确保消息不超过最大token数量
trimmer = trim_messages(
    max_tokens=4096,
    strategy="last",
    token_counter=tiktoken_counter,
    include_system=True,
)

# 存储会话历史记录
store = {}

# 获取会话历史记录
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 创建OpenAI模型
model = ChatOpenAI()

# 创建提示模板，参考https://smith.langchain.com/hub/rlm/rag-prompt
"""
    在这段提示词里，我们告诉大模型，根据提供的上下文回答问题，不知道就说不知道。这是一个提示词模板，
    在提示词的最后是我们给出的上下文（Context）。这里上下文是根据问题检索出来的内容。有了这个提示词，
    再加上聊天历史和我们的问题，就构成了一个完整的提示词模板
"""
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Context: {context}""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# 格式化检索到的文档
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 组装各个组件，构成一条链
# 首先构建了一个 context 变量，它也一条链。第一步是从传入参数中获取到 question 属性，也就是我们的问题，
# 然后把它传给 retriever。retriever 会根据问题去做检索，对应到我们这里的实现，就是到向量数据库中检索，检索的结果是一个文档列表
# 文档是 LangChain 应用内部的表示，要传给大模型，我们需要把它转成文本，这就是 format_docs 做的事情，它主要是把文档内容取出来拼接到一起
"""
    RunnablePassthrough.assign 这个函数就是在不改变链当前状态值的前提下，添加新的状态值。
    前面我们说了，这里赋给 context 变量的值是一个链，我们可以把它理解成一个函数，它会在运行期执行，
    其参数就是我们当前的状态值。现在你可以理解 itemgetter(“question”) 的参数是从哪来的了。
    这个函数的返回值会用来在当前的状态里添加一个叫 context 的变量，以便在后续使用。
"""
context = itemgetter("question") | retriever | format_docs
first_step = RunnablePassthrough.assign(context=context)
chain = first_step | prompt | trimmer | model

# 创建一个 RunnableWithMessageHistory 对象，它会在当前的链上添加一个会话历史记录。
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# 配置会话ID
config = {"configurable": {"session_id": "dreamhead"}}

# 主循环，等待用户输入
while True:
    user_input = input("You:> ")
    if user_input.lower() == 'exit':
        break

    if user_input.strip() == "":
        continue

    stream = with_message_history.stream(
        {"question": user_input},
        config=config
    )
    for chunk in stream:
        print(chunk.content, end='', flush=True)
    print()