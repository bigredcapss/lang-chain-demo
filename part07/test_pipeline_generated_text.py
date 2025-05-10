# -*- coding: UTF-8 -*-
# filename : test_pipeline_generated_text.py
# description : 使用高层接口调用hugging face上的模型生成文本
# author by : peanut
# date : 2025/5/5

"""
    高层接口相当于是对底层接口做了封装，让代码使用起来更容易，相对而言，底层接口则有更大的灵活性。
    在高层接口里，我们核心要知道的一个概念就是管道（pipeline）。和软件领域所有叫管道的概念一样，
    它要做的就是一步一步地进行处理，一个阶段完成之后，交给下一个阶段
    在文本生成这个场景下，管道要做的就是：
        1. 对输入的文本进行分词
        2. 将分词后的文本转换为模型需要的输入格式
        3. 将输入格式交给模型进行处理
        4. 将模型输出的结果转换为文本
        5. 返回文本


    第一次执行这段代码可能会遇到很多问题。有的模型在访问上是需要受到限制的，比如，Meta 的 Llama，
    需要我们先去做相应的申请。我们要有 Hugging Face 账户，再去做对应的申请，
    然后，还需要配置自己的 Access Token，再去做相应的配置
"""


import torch
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

messages = [
    {"role": "user", "content": "请写一首赞美秋天的五言绝句"},
]
# 构建了一个管道，第一个参数指定了它的用途，这里是文本生成（text-generation），
#  pipeline 会根据不同的用途进行不同的管道配置。第二个参数是模型，在这个例子里面，
# 我们使用的模型是阿里的通义千问（Qwen），引用模型的方式就是“用户名 / 模型名”，在这里就是“Qwen/Qwen2.5-0.5B-Instruct”
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device=device, max_new_tokens=100)
result = pipe(messages)
print(result[-1]['generated_text'][-1]['content'])