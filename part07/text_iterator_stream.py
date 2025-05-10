# -*- coding: UTF-8 -*-
# filename : text_iterator_stream.py
# description : 使用文本迭代器流式处理生成文本
# author by : peanut
# date : 2025/5/5


from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, TextIteratorStreamer


"""
    与之前最大的不同是，这段代码启用了多线程，这样一来，生成和输出是异步处理的，不会彼此阻塞，
    更符合真实代码中的处理。正如 TextIteratorStreamer 这个名字所显示的，
    它实现了 Iterator，所以，我们可以在其上进行迭代
    
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
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=20)
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

for text in streamer:
    print(text)