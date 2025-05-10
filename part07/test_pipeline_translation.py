# -*- coding: UTF-8 -*-
# filename : test_pipeline_translation.py
# description : 使用高层接口调用hugging face上的模型进行翻译
# author by : peanut
# date : 2025/5/5

import torch
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en", device=device)
result = pipe("今天天气真好，我想出去玩。")
print(result[-1]['translation_text'])