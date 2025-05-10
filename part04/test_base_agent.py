# -*- coding: UTF-8 -*-

# filename : test_base_agent.py
# description : 基于OpenAI+ReAct实现基础Agent
# author by : peanut
# date : 2025/5/4

import re
from openai import OpenAI


DEFAULT_MODEL = "gpt-4o-mini"
client = OpenAI()

# 基础的聊天机器人
class Agent:
    """
        初始化的时候，我们可以传入系统提示词给这个 Agent 做一些初始的设定。 messages 是我们维护的历史消息列表，
        如果设定了系统提示词，就把它加到历史消息里。
    """
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})


    """
        invoke 是主要的对外接口。调用之前，我们把消息存放到历史消息里，等到调用大模型之后，再把应答存放到历史消息里
    """
    def invoke(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    """
        execute 处理了请求大模型的过程，模型和消息都比较好理解。temperature 的值是 0，因为这里用到的是大模型的推理过程，
        所以，我们希望确定性强一些。另外，这里我们使用了同步处理，因为我们这里是把大模型当作推理引擎，需要等待所有的内容回来。
    """
    def execute(self):
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=self.messages,
            temperature=0
        )
        return completion.choices[0].message.content
    

"""
    这段提示词是 ReAct 的典型提示词，它告诉大模型，在思考、行动、暂停、观察的循环中运行。最后，大模型会输出一个答案。
    ReAct 描述，给大模型解释了 ReAct 的三个阶段：思考（Thought）、行动（Action）、观察（Observation）。
    这里还多了一个暂停（Pause），其主要的目的就是停下来，这时执行流程就回到我们这里，执行相应的动作。
    当动作执行完毕，再把控制权返回给大模型
"""
# 基于ReAct的提示词
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

ask_fruit_unit_price:
e.g. ask_fruit_unit_price: apple
Asks the user for the price of a fruit

Example session:

Question: What is the unit price of apple?
Thought: I need to ask the user for the price of an apple to provide the unit price. 
Action: ask_fruit_unit_price: apple
PAUSE

You will be called again with this:

Observation: Apple unit price is 10/kg

You then output:

Answer: The unit price of apple is 10 per kg.
""".strip()


# 根据ReAct提示词 实现对应的动作
def calculate(what):
    return eval(what)


def ask_fruit_unit_price(fruit):
    if fruit.casefold() == "apple":
        return "Apple unit price is 10/kg"
    elif fruit.casefold() == "banana":
        return "Banana unit price is 6/kg"
    else:
        return "{} unit price is 20/kg".format(fruit)
    

# 组合实现Agent
action_re = re.compile(r'^Action: (\w+): (.*)$')

known_actions = {
    "calculate": calculate,
    "ask_fruit_unit_price": ask_fruit_unit_price
}


# 实现query 函数，其核心是一个循环。每次循环我们都会询问大模型，
# 起始的提示词就是用户的问题。每次询问的结果，我们会先从里面分析出要执行的动作，
# 这里采用了正则表达式直接匹配文本，如果匹配到，就说明有动作要执行。
def query(question, max_turns=5):
    i = 0
    agent = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = agent.invoke(next_prompt)
        print(result)
        actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return
        
