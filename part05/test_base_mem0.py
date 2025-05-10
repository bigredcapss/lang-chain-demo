# -*- coding: UTF-8 -*-

# filename : test_base_mem0.py
# description : mem0-大模型长期记忆的解决方案
# author by : peanut
# date : 2025/5/4

"""
    大模型长期记忆的解决方案-https://github.com/mem0ai/mem0

    我们再来看配置。我们配置了大模型、Embedding 模型，还有向量数据库。
    对于长期记忆的搜索需要基于语义，所以，这里配置 Embedding 模型和向量数据库是很容易理解的。
    
    但为什么还要配置大模型呢？因为 mem0 并不是把数据直接存到向量数据库里的。
    调用 add 时，mem0 会先把内容发送给大模型，让大模型从内容中提取出一些事实（fact），
    真正存放到向量数据库里的实际上是这些事实。

"""


from mem0 import Memory

config = {
    "version": "v1.1",
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0,
            "max_tokens": 1500,
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-ada-002"
        }
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "mem0db",
            "path": "mem0db",
        }
    },
    "history_db_path": "history.db",
}

m = Memory.from_config(config)

m.add("我喜欢读书", user_id="peanut", metadata={"category": "hobbies"})
m.add("我喜欢编程", user_id="peanut", metadata={"category": "hobbies"})

related_memories = m.search(query="peanut有哪些爱好？", user_id="peanut")
print(' '.join([mem["memory"] for mem in related_memories['results']]))



