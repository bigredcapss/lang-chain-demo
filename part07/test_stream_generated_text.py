# -*- coding: UTF-8 -*-
# filename : test_stream_generated_text.py
# description : 使用流式处理生成文本
# author by : peanut
# date : 2025/5/5

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


"""
    在这个例子里，我们用到了 TextStreamer，它会直接把生成结果输出到控制台上。
    如果我们要实现一个控制台应用，它是可以用的。但更多的情况下，我们需要拿到输出结果，再去做相应的处理
    
"""

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

messages = [
    {"role": "user", "content": "请写一首赞美秋天的五言绝句"},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    streamer=streamer,
)