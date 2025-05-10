# -*- coding: UTF-8 -*-

# filename : test_langchain_agent.py
# description : 基于LangChain实现Agent
# author by : peanut
# date : 2025/5/4


from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


"""
    @tool 是一个装饰器，它让我们把一个函数变成了一个工具（tool）。
    工具在 LangChain 里是一个重要的概念，它和我们说的 Agent 系统架构中的工具概念是可以对应上的，
    工具主要负责执行查询，或是完成一个一个的动作。

    Agent 在执行过程中，会获取工具的信息，传给大模型。这些信息主要就是一个工具的名称、描述和参数，
    这样大模型就知道该在什么情况下怎样调用这个工具了。@tool 可以提取函数名变成工具名，提取参数变成工具的参数，
    还有一点就是，它可以提取函数的 Docstring 作为工具的描述。这样一来，calculate 就从一个普通的函数变成了一个工具。

"""


@tool
def calculate(what: str) -> float:
    """Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary"""
    return eval(what)

@tool
def ask_fruit_unit_price(fruit: str) -> str:
    """Asks the user for the price of a fruit"""
    if fruit.casefold() == "apple":
        return "Apple unit price is 10/kg"
    elif fruit.casefold() == "banana":
        return "Banana unit price is 6/kg"
    else:
        return "{} unit price is 20/kg".format(fruit)


"""
    提示词模板
    作为一个模板，这里面有几个空是留给我们的，最主要的就是 tools 和 tool_names 两个变量，这就是工具的信息。
    tool_names 很简单，就是工具的名称。tools 是工具格式化成一个字符串。
    比如，在缺省的实现中， calculate 就会格式化成下面这个样子，可以看到它包括了工具的基本属性都拼装了进去。
    calculate(what: str) -> float - Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary
    input 也比较好理解，就是我们的输入。 
    agent_scratchpad 是在 Agent 的执行过程中，存放中间过程的，你可以把它理解成我们上一讲的聊天历史部分
    
    
"""

prompt = PromptTemplate.from_template('''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}''')


"""
    组装agent
    我们调用 create_react_agent 创建了一个基于 ReAct 的 Agent。前面说过，ReAct 的 Agent 能够正常运行，需要提示词与代码配合起来使用。
    我前面给出的提示词就是要与 create_react_agent 函数配合在一起使用的。
    
    create_react_agent 完成的工作就是基于这段提示词的执行过程进行处理，比如，
    解析返回内容中的动作（Action）与动作输入（Action Input），还有前面说的 agent_scratchpad 的处理过程，也是在这个函数中组装进去的。
    
    站在软件设计的角度看，二者结合如此紧密，却被分开了，等于破坏了封装。
    实际上，二者之前确实是合在一起的，就是一个 create_react_agent 函数。
    现在将二者分开，是为了给使用者一个调整提示词的机会。
    
    
    与之前几个基于 LangChain 的应用最大的不同在于，我们这个 Agent 的实现并没有组装成一个链。
    正如我们前面所说，Agent 的核心是一个循环，这其实是一个流程，而之前的应用从头到尾都是一个“链”式过程。
    所以，这里用到了 AgentExecutor。
    
    即便不看它的实现，你应该也能知道，其核心实现就是一个循环：判断是不是该结束，不是的话，继续下一步，向大模型发送消息，是的话，跳出循环。
"""
tools = [calculate, ask_fruit_unit_price]
model = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


"""
    执行agent
"""


result = agent_executor.invoke({
    "input": "What is the total price of 3 kg of apple and 2 kg of banana?"
})
print(result)