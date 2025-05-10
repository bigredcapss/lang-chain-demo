# -*- coding: UTF-8 -*-
# filename : test_langchain_huggine_face_demo.py
# description : 使用 langchain 调用 hugging face 上的模型
# author by : peanut
# date : 2025/5/5


import torch
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace


"""
    使用 langchain 调用 hugging face 上的模型
    LangChain，重点在于它提供的不同抽象，可以帮助搭建各种大模型应用。其中LangChain 最基础的抽象——模型。

    LangChain 的模型就是 LangChain 给我们提供的一层封装，屏蔽掉了不同大模型之间的差异，让我们可以方便地在不同大模型之间进行切换。
    任何想要接入 LangChain 体系的大模型，只要实现了相应的接口，就可以无缝地嵌入到 LangChain 的体系中去，Hugging Face 的模型就是这么做的。
    
    我们之所以要把 Hugging Face 模型嵌入到 LangChain 的体系中，主要是因为我们希望使用 LangChain 提供的其它抽象。
    要使用 Hugging Face 相关的代码，首先需要安装相应的包：
    pip install langchain-huggingface

    
    langchain-huggingface 是 Hugging Face 和 LangChain  共同维护的一个包，
    其目标是缩短将 Hugging Face 生态的新功能带给 LangChain 用户的时间。它里面包含了很多功能：
        有各种模型的实现，比如聊天模型和 Embedding 模型；
        有数据集的实现，它实现成了 DocumentLoader；
        有工具的实现，比如文本分类、文本转语音等。

    从提供的内容上来看，这个包封装了 Hugging Face 上的主要能力——模型和数据集。
    其中，不同的模型因为能力上的差异做了不同归结：属于大语言模型的，就归结到了 LangChain 模型上，而无法归结的，就以工具的形式提供
    
"""
device = 0 if torch.cuda.is_available() else -1

# 构建管道,HuggingFacePipeline 是一个 LangChain 的 LLM 类型
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    device=device,
    pipeline_kwargs=dict(
        max_new_tokens=512,
        return_full_text=False,
    ),
)

# 封装成ChatModel
chat_model = ChatHuggingFace(llm=llm)

# 在 LangChain 的体系下，有 ChatModel 和 LLM 两种抽象都可以处理大模型。
# 我们这里定义出的 HuggingFacePipeline 就是 LLM，它也完全可以拿过来单独使用
# llm.invoke("写一首赞美秋天的五言绝句。")
result = chat_model.invoke("写一首赞美秋天的五言绝句。")
print(result.content)