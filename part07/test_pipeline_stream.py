# -*- coding: UTF-8 -*-
# filename : test_pipeline_stream.py
# description : 给 pipeline 增加流式输出的能力
# author by : peanut
# date : 2025/5/5

"""
    给 pipeline 增加流式输出的能力
"""


import torch
from threading import Thread
from transformers import pipeline, TextIteratorStreamer

device = "cuda" if torch.cuda.is_available() else "cpu"

messages = [
    {"role": "user", "content": "请写一首赞美秋天的五言绝句"},
]

pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device=device, max_new_tokens=100)

streamer = TextIteratorStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)

generation_kwargs = dict(text_inputs=messages, streamer=streamer)
thread = Thread(target=pipe, kwargs=generation_kwargs)
thread.start()

for text in streamer:
    print(text)