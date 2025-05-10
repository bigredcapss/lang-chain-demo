# -*- coding: UTF-8 -*-

# filename : test_langchain_tools_agent.py
# description : 基于LangChain的工具实现Agent
# author by : peanut
# date : 2025/5/4


from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import ShellTool
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor

prompt = ChatPromptTemplate.from_template('''Answer the following questions as best you can. You have access to the following tools:

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

shell_tool = ShellTool()

tools = [shell_tool]
model = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({
    "input": "Create a file named 'test.txt' and write 'Hello, World!' to it."
})