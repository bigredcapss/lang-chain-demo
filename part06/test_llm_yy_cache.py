# -*- coding: UTF-8 -*-

# filename : test_llm_yy_cache.py
# description : 大模型语义缓存
# author by : peanut
# date : 2025/5/4


"""

    大模型应用的特点就决定了精确缓存往往是失效的。因为大模型应用通常采用的是自然语言交互，
    以自然语言为提示词，就很难做到完全相同。像前面test_llm_cache.py中展示的那个例子，实际上是特意构建的，
    才能保证精确匹配。所以，语义匹配就成了更好的选择。

    语义匹配我们并不陌生，前面讲 RAG 时，我们讲过了基于向量的语义匹配。LangChain 社区提供了许多语义缓存的实现，
    在各种语义缓存中，我们最熟悉的应该是 Redis。
    
    在大部分人眼中，Redis 应该属于精确匹配的缓存。Redis 这么多年也在不断地发展，有很多新功能不断地拓展出来，
    最典型的就是 Redis Stack，它就是在原本开源 Redis 基础上扩展了其它的一些能力。
    
    比如，对 JSON 支持（RedisJSON），对全文搜索的支持（RediSearch），对时序数据的支持（RedisTimeSeries），
    对概率结构的支持（RedisBloom）。其中，支持全文搜索的 RediSearch 就可以用来实现基于语义的搜索。
    全文搜索，本质上也是语义搜索，而这个能力刚好就是语义缓存中需要的。

    Redis 对于语义缓存的支持是基于 RediSearch 的。所以，要想使用语义缓存，
    我们需要使用安装了 RediSearch 的 Redis，一种方式是使用 Redis Stack
"""

from langchain.globals import set_llm_cache
from langchain_community.cache import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import Sequence, Optional, Any
from langchain_core.outputs import Generation
from langchain.cache import BaseCache
import json
import time

RETURN_VAL_TYPE = Sequence[Generation]

def prompt_key(prompt: str) -> str:
    messages = json.loads(prompt)
    result = ["('{}', '{}')".format(data['kwargs']['type'], data['kwargs']['content']) for data in messages if
               'kwargs' in data and 'type' in data['kwargs'] and 'content' in data['kwargs']]
    return ' '.join(result)


class FixedSemanticCache(BaseCache):
    def __init__(self, cache: BaseCache):
        self.cache = cache

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        key = prompt_key(prompt)
        return self.cache.lookup(key, llm_string)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        key = prompt_key(prompt)
        return self.cache.update(key, llm_string, return_val)

    def clear(self, **kwargs: Any) -> None:
        return self.cache.clear(**kwargs)

# docker run -p 6379:6379 redis/redis-stack-server:latest 
set_llm_cache(
    FixedSemanticCache(
        RedisSemanticCache(redis_url="redis://localhost:6379",
                           embedding=OpenAIEmbeddings())
    )
)

model = ChatOpenAI(model="gpt-4o-mini")

start_time = time()
response = model.invoke("""请给我讲一个一句话笑话""")
end_time = time()
print(response.content)
print(f"第一次调用耗时: {end_time - start_time}秒")

start_time = time()
response = model.invoke("""你能不能给我讲一个一句话笑话""")
end_time = time()
print(response.content)
print(f"第二次调用耗时: {end_time - start_time}秒")



