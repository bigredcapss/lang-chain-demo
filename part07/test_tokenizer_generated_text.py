# -*- coding: UTF-8 -*-
# filename : test_tokenizer_generated_text.py
# description : 使用底层接口调用hugging face上的模型生成文本
# author by : peanut
# date : 2025/5/5

from transformers import AutoTokenizer, AutoModelForCausalLM

# 初始化 Tokenizer 和模型
# 过程中涉及到了 Token 和文本之间的转换，所以，这里还有一个 Tokenizer，它就是负责转换的模型。
# 一般来说，大模型都要和对应的 Tokenizer 一起使用，所以，你会看到它俩往往给出的是同一个模型名字
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

messages = [
    {"role": "user", "content": "请写一首赞美春天的诗，要求不包含春字"},
]

# 第一步，把输入转换成 Token
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 第二步，大模型根据输入生成相应的内容。
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)

# 第三步，生成的结果是 Token，还需要把它转成文本。
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)