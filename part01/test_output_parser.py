# -*- coding: UTF-8 -*-

# filename : test_output_parser.py
# description : LangChain的输出解析器OutputParser抽象-->具体实现为StrOutputParser、JsonOutputParser
# author by : peanut
# date : 2025/5/1


"""
    1、LangChain 的 OutputParser 抽象，它负责将模型输出的内容转换为特定的格式。
    2、LangChain 里提供了一些常用的解析器，比如，StrOutputParser，它负责将模型输出的内容转换为字符串。
    3、LangChain 还提供了许多不同的输出格式解析器，比如：JSON、CSV、分割符、枚举等

"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
import os

# 从环境变量获取配置
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"  # 替换为您的 API 密钥
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"

# 提示词模版
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Translate the following from English into Chinese:"),
        ("user", "{text}")
    ]
)

# 模型
model = ChatOpenAI(model="gpt-4o-mini")

# 输出解析器
output_parser = StrOutputParser()

# 通过 LCEL 把 prompt_template 和 model 和 output_parser 组成一条链
chain = prompt_template | model | output_parser

# 触发链式调用
result = chain.invoke({"text":"Welcome to LLM application development!"})

print(result)



# 声明大模型返回的格式
class Work(BaseModel):
    title: str = Field(description="Title of the work")
    description: str = Field(description="Description of the work")

# 声明输出解析器
parser = JsonOutputParser(pydantic_object=Work)

prompt = PromptTemplate(
    template="列举3部{author}的作品。\n{format_instructions}",
    input_variables=["author"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

model = ChatOpenAI(model="gpt-4o-mini")

chain = prompt | model | parser
result = chain.invoke({"author": "老舍"})
print(result)