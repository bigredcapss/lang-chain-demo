# -*- coding: UTF-8 -*-

# filename : test_rag_index.py
# description : 基于Chroma实现RAG索引化过程
# author by : peanut
# date : 2025/5/1


"""

LangChain 支持了很多的向量数据库，它们都有一个统一的接口：VectorStore，在这个接口中包含了向量数据库的统一操作，
比如添加、查询之类的。这个接口屏蔽了向量数据库的差异，在向量数据库并不为所有程序员熟知的情况下，
给尝试不同的向量数据库留下了空间。各个具体实现负责实现这些接口，我们这里采用的实现是 Chroma。

在 Chroma 初始化的过程中，我们指定了 Embedding 函数，它负责把文本变成向量。这里我们采用了 OpenAI 的 Embeddings 实现，
你完全可以根据自己的需要选择相应的实现，LangChain 社区同样提供了大量的实现，
比如，你可以指定 Hugging Face 这个模型社区中的特定模型来做 Embedding。



"""

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 第一步：加载文档
loader = TextLoader("./introduce.txt")
docs = loader.load()

# 第二步：分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 第三步：创建向量存储
vectorstore = Chroma(
    collection_name="ai_learning",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="vectordb"
)
# 第四步：添加文档到向量存储
vectorstore.add_documents(splits)

# 第五步：调用 similarity_search 检索向量数据库的数据
documents = vectorstore.similarity_search("文章的作者是谁？")
print(documents)








