# -*- coding: UTF-8 -*-

# filename : test_llm_cache.py
# description : 基于langchain提供的抽象实现大模型精确缓存
# author by : peanut
# date : 2025/5/4

from time import time

from langchain.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_openai import ChatOpenAI


"""
    基于langchain提供的抽象实现大模型缓存
    在 LangChain 里，缓存是一个全局选项，只要设置了缓存，所有的大模型都可以使用它。

    如果某个特定的大模型不需要缓存，可以在设置的时候关掉缓存，e.g
    model = ChatOpenAI(model="gpt-4o-mini", cache=False)

    当然，如果你不想缓存成为一个全局选项，只想针对某个特定进行设置也是可以的：
    model = ChatOpenAI(model="gpt-4o-mini", cache=InMemoryCache())

    LangChain 里的缓存是一个统一的接口，其核心能力就是把生成的内容插入缓存以及根据提示词进行查找。
    LangChain 社区提供了很多缓存实现，像我们在前面例子里用到的内存缓存，还有基于数据库的缓存，当然，也有我们最熟悉的 Redis 缓存。

    虽然 LangChain 提供了许多缓存实现，但本质上说，只有两类缓存——精确缓存和语义缓存。
    精确缓存，只是在提示词完全相同的情况下才能命中缓存，它和我们理解的传统缓存是一致的，我们前面用来演示的内存缓存就是精确缓存。

"""


# 设置缓存
set_llm_cache(InMemoryCache())

# 创建大模型
model = ChatOpenAI(model="gpt-4o-mini")

# 第一次调用
start_time = time()
response = model.invoke("给我讲个一句话笑话")
end_time = time()
print(response.content)
print(f"第一次调用耗时: {end_time - start_time}秒")

# 第二次调用
start_time = time()
response = model.invoke("给我讲个一句话笑话")
end_time = time()
print(response.content)
print(f"第二次调用耗时: {end_time - start_time}秒")