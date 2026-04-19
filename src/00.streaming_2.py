#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-02-27
# @function: 流式输出
# @version : V0.5
# @Description ：测试流式输出。未成功！

"""
这个脚本尝试实现AI智能体（Agent）的流式输出功能。

核心概念：
什么是"流式输出"（Streaming）？
→ 传统模式：等待AI生成完整回答后，一次性返回所有内容
→ 流式模式：AI每生成一个字/词，就立即返回，实现"打字机"效果

实际应用场景：
- 聊天界面：用户可以看到AI正在逐字生成回答，而不是等待很久
- 长文本生成：减少用户等待焦虑，提升体验
- 实时反馈：可以实时监控AI的思考过程

本脚本的目标：
1. 创建一个带有工具调用能力的AI智能体
2. 实现流式输出，实时显示AI的回答
3. 探索不同的流式输出方法

注意：根据文件头注释"未成功"，说明这些方法可能存在问题或需要调整。
"""

# ==================== 导入必要的库 ====================

# 导入随机数模块，用于模拟猫藏身位置的随机选择
import random

# 导入系统模块，用于直接控制标准输出
import sys

# 从 langchain_classic 导入智能体相关类
# AgentExecutor: 智能体执行器，负责运行智能体并管理工具调用
# create_tool_calling_agent: 创建支持工具调用的智能体的工厂函数
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

# 导入回调处理器，用于处理流式输出
# FinalStreamingStdOutCallbackHandler: 专门用于最终答案的流式输出
from langchain_classic.callbacks import FinalStreamingStdOutCallbackHandler

# 导入基础回调处理器和流式输出回调处理器
# BaseCallbackHandler: 所有回调处理器的基类
# StreamingStdOutCallbackHandler: 标准的流式输出处理器，将token直接打印到控制台
from langchain_core.callbacks import BaseCallbackHandler, StreamingStdOutCallbackHandler

# 导入工具装饰器，用于将普通函数转换为AI可调用的工具
from langchain_core.tools import tool

# 导入Ollama聊天模型接口
from langchain_ollama import ChatOllama

# ==================== 配置模型名称 ====================

# 定义要使用的模型名称
llm_model_name = "qwen3.5"


# 使用阿里云的通义千问 Qwen 3.5 模型
# 这是一个强大的中文大语言模型，支持工具调用


# ==================== 定义工具函数 ====================

@tool
def where_cat_is_hiding() -> str:
    """
    查询猫当前藏在哪里

    这是一个简单的工具函数，模拟查询猫的藏身位置。

    装饰器 @tool 的作用：
    → 将普通Python函数转换为LangChain工具
    → AI可以识别并调用这个函数
    → 函数的文档字符串（docstring）会告诉AI这个工具的用途

    返回值：
        str: 猫的藏身位置，随机返回"床下"或"架子上"
    """
    # 随机选择猫的位置
    return random.choice(["under the bed", "on the shelf"])


@tool
def get_items(place: str) -> str:
    """
    查询指定位置有哪些物品

    使用这个工具来查找给定位置中有哪些物品。

    参数：
        place (str): 要查询的位置，例如 "bed"、"shelf"

    返回值：
        str: 该位置的物品列表，用逗号分隔

    工作原理：
    - 如果位置包含 "bed"（床），返回床下的物品
    - 如果位置包含 "shelf"（架子），返回架子上的物品
    - 其他位置，返回猫零食（默认值）
    """

    # 检查位置是否包含 "bed"（床）
    if "bed" in place:  # For under the bed
        # 床下的物品：袜子、鞋子和灰尘球
        return "socks, shoes and dust bunnies"

    # 检查位置是否包含 "shelf"（架子）
    if "shelf" in place:  # For 'shelf'
        # 架子上的物品：书籍、铅笔和照片
        return "books, penciles and pictures"

    # 如果查询其他位置
    else:  # if the agent decides to ask about a different place
        # 返回猫零食（默认值）
        return "cat snacks"


# ==================== 创建提示词模板 ====================

"""
2. 初始化智能体
"""

# 导入聊天提示词模板类
from langchain_core.prompts import ChatPromptTemplate

# 创建聊天提示词模板
# 这个模板定义了AI智能体的行为准则和输入格式

prompt = ChatPromptTemplate.from_messages([
    # 第一条消息：系统指令
    # 设定AI的角色和行为准则
    ("system", "You are a helpful assistant"),
    # 系统提示："你是一个乐于助人的助手"

    # 第二条消息：聊天历史占位符
    # 用来放置之前的对话历史
    ("placeholder", "{chat_history}"),
    # {chat_history} 会被替换成实际的对话历史
    # placeholder 表示这个位置可能有内容，也可能为空

    # 第三条消息：用户输入
    # 这是用户当前提出的问题
    ("human", "{input}"),
    # {input} 会被替换成用户的实际问题
    # 例如："猫藏在哪里？那里有什么物品？"

    # 第四条消息：智能体思考过程占位符
    # 用来记录智能体的思考和工具调用过程
    ("placeholder", "{agent_scratchpad}"),
    # {agent_scratchpad} 是智能体的"草稿纸"
    # 记录智能体决定调用哪些工具、工具返回什么结果等
])

# 定义工具列表
# 将之前定义的两个工具函数放入列表中
tools = [get_items, where_cat_is_hiding]


# AI可以根据需要选择调用这些工具来获取信息


# ==================== 自定义流式输出处理器1 ====================

class CustomStreamingHandler(FinalStreamingStdOutCallbackHandler):
    """
    自定义流式输出处理器 - 方法1

    继承自 FinalStreamingStdOutCallbackHandler
    → 专门用于处理最终答案的流式输出

    工作原理：
    每当AI生成一个新的token（字/词片段），就会调用 on_llm_new_token 方法
    我们将这些token收集起来并立即打印出来
    """

    def __init__(self):
        """
        初始化方法

        创建一个缓冲区（buffer）用来存储生成的token
        """
        # 调用父类的初始化方法
        super().__init__()

        # 创建一个空列表作为缓冲区
        # 用来存储AI生成的所有token
        self.buffer = []

    def on_llm_new_token(self, token: str, **kwargs):
        """
        每当 LLM 生成新 token 时自动调用这个方法

        这是流式输出的核心！

        参数：
            token (str): AI新生成的文本片段
                例如："你"、"好"、"！"
            **kwargs: 其他额外参数（暂时不用）

        工作流程：
        1. 将新token添加到缓冲区
        2. 立即打印token到控制台
        3. 使用 end="^" 让每个token后面显示 ^ 符号（便于观察）
        4. 使用 flush=True 强制立即输出，不等待缓冲区满
        """

        # 将新token添加到缓冲区列表中
        self.buffer.append(token)

        # 直接打印token，实现流式输出
        # end="^": 每个token后面添加 ^ 符号，方便看到每个token的边界
        # flush=True: 强制立即输出到控制台，不等待缓冲区
        print(token, end="^", flush=True)  # 直接流式输出最终结果


# ==================== 自定义流式输出处理器2 ====================

class CallbackHandler(StreamingStdOutCallbackHandler):
    """
    自定义输出处理器 - 方法2

    继承自 StreamingStdOutCallbackHandler
    → 标准的流式输出处理器

    这个处理器更复杂，尝试只提取"Final Answer"部分的内容
    目的是过滤掉智能体思考过程的输出，只显示最终答案

    工作原理：
    1. 累积所有生成的token
    2. 检测是否进入"Final Answer"阶段
    3. 只输出最终答案部分的内容
    """

    def __init__(self):
        """
        初始化方法

        设置两个状态变量来跟踪输出进度
        """

        # 创建一个字符串变量，用来累积所有的token内容
        self.content: str = ""

        # 布尔标志，标记是否进入了"Final Answer"阶段
        # False: 还在思考或调用工具阶段
        # True: 已经开始输出最终答案
        self.final_answer: bool = False

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        """
        每当 LLM 生成新 token 时自动调用这个方法

        智能体会逐渐返回json格式的结果，这里只输出 action_input 的内容

        参数：
            token (str): AI新生成的文本片段
            **kwargs: 其他额外参数

        工作流程：
        1. 将token累积到content中
        2. 检测是否出现"Final Answer"标记
        3. 如果进入了最终答案阶段，提取并输出相关内容
        """

        # 将新token累积到content字符串中
        self.content += token

        # 检查累积的内容中是否包含 "Final Answer" 标记
        # "Final Answer" 是智能体准备输出最终答案时的标记
        if "Final Answer" in self.content:
            # 现在我们到了 Final Answer 部分，但不要立即打印
            # 将标志设置为True，表示后续内容需要处理
            self.final_answer = True

            # 清空content，重新开始累积最终答案部分的内容
            self.content = ""

        # 如果已经进入最终答案阶段
        if self.final_answer:
            # 检查是否包含 '"action_input": "' 这个JSON键
            # 这表示智能体正在输出某个动作的输入参数
            if '"action_input": "' in self.content:

                # 当token中包含 '}' 时，移除 '}' 和后面的字符
                # 因为 '}' 表示JSON对象结束，后面的内容不需要

                # 找到 '}' 的位置索引
                index = token.find('}')  # 找到 '}' 的索引

                # 如果找到了 '}'（index不等于-1）
                if index != -1:
                    # 退出最终答案阶段
                    self.final_answer = False

                    # 截取token，只保留 '}' 之前的部分
                    token = token[:index]

                # 将处理后的token写入标准输出
                sys.stdout.write(token)

                # 如果没有找到 '}'，在token后面添加 ^ 符号作为标记
                if index == -1:
                    sys.stdout.write('^')

                # 强制刷新输出缓冲区，确保内容立即显示
                sys.stdout.flush()

    def on_llm_end(self, response, **kwargs):
        """
        当 LLM 完成所有token生成时调用这个方法

        这是清理和收尾工作的地方

        参数：
            response: LLM的完整响应对象
            **kwargs: 其他额外参数

        返回值：
            调用父类的 on_llm_end 方法
        """
        # 调用父类的方法，执行标准的结束处理
        return super().on_llm_end(response, **kwargs)


# ==================== 创建AI模型实例 ====================

# 创建ChatOllama模型实例
# 配置模型参数和回调处理器

model = ChatOllama(
    model=llm_model_name,  # 使用 qwen3.5 模型
    temperature=0.3,  # 温度值，控制创造性（0.3比较保守）
    verbose=True,  # 开启详细模式，显示调试信息
    callbacks=[StreamingStdOutCallbackHandler()]  # 添加标准流式输出回调
)
# callbacks 参数：
# → 传入一个回调处理器列表
# → StreamingStdOutCallbackHandler() 会自动将每个token打印到控制台
# → 这是最简单的流式输出方法


# ==================== 创建智能体 ====================

# 使用 create_tool_calling_agent 函数创建智能体
# 这个智能体具备调用工具的能力

agent = create_tool_calling_agent(model, tools, prompt)
# 参数说明：
# 1. model: AI模型实例（ChatOllama）
# 2. tools: 工具列表（[get_items, where_cat_is_hiding]）
# 3. prompt: 提示词模板
#
# 返回：
# 一个可以调用工具的智能体对象


# ==================== 创建智能体执行器 ====================

# 创建 AgentExecutor（智能体执行器）
# 它负责运行智能体、管理工具调用、处理错误等

agent_executor = AgentExecutor(agent=agent, tools=tools).with_config(
    {"run_name": "Agent"}
)


# 参数说明：
# 1. agent: 刚才创建的智能体对象
# 2. tools: 工具列表
#
# .with_config({"run_name": "Agent"}):
# → 为这个执行器设置一个名称 "Agent"
# → 在日志和回调中会显示这个名称
# → 方便调试和追踪


# ==================== 函数1：咨询智能体（基础版本） ====================

def ask(question):
    """
    咨询智能体 - 基础版本

    使用 stream() 方法逐步获取智能体的执行结果

    参数：
        question (str): 用户的问题
            例如："猫藏在哪里？那里有什么物品？"

    工作流程：
    1. 调用 agent_executor.stream() 方法
    2. 逐个接收智能体返回的数据块（chunk）
    3. 调用 print_normal() 打印每个数据块

    注意：
    这个方法不是真正的流式输出token
    而是流式输出智能体的执行步骤（调用工具、获取结果等）
    """

    # 使用 for 循环遍历智能体的流式输出
    # stream() 方法返回一个迭代器，每次yield一个数据块

    for chunk in agent_executor.stream({"input": question}):
        # chunk 是智能体执行过程中的一个数据块
        # 可能包含：
        # - actions: 智能体决定执行的动作
        # - steps: 工具执行的结果
        # - output: 最终的输出结果

        # 调用 print_normal 函数打印数据块
        print_normal(chunk)


# ==================== 打印函数1：普通打印 ====================

def print_normal(chunk):
    """
    普通打印函数

    直接打印数据块，用于调试和查看原始数据结构

    参数：
        chunk: 智能体返回的数据块（字典类型）
    """

    # 打印整个数据块
    print(chunk)

    # 打印分隔线，便于区分不同的数据块
    print("----")


# ==================== 打印函数2：简化打印 ====================

def print_simple(chunk):
    """
    简化打印函数

    使用 pprint 模块以更易读的格式打印数据块
    限制深度为1，只显示顶层结构

    参数：
        chunk: 智能体返回的数据块
    """

    # 导入 pprint 模块（pretty-print）
    # pprint 可以更美观地打印复杂的数据结构
    import pprint

    # 打印分隔线
    print("----")

    # 使用 pprint 打印数据块
    # depth=1: 只显示第一层结构，不展开嵌套的内容
    # 这样更容易从宏观上理解数据结构
    pprint.pprint(chunk, depth=1)


# ==================== 打印函数3：有用信息打印 ====================

def print_useful(chunk):
    """
    有用信息打印函数

    解析数据块，提取并打印关键信息
    包括：调用的工具、工具结果、最终输出

    参数：
        chunk: 智能体返回的数据块

    工作流程：
    1. 检查数据块类型（actions、steps、output）
    2. 根据类型提取并打印相关信息
    3. 如果类型未知，抛出异常
    """

    # 情况1：数据块包含 "actions" 键
    # 表示智能体决定执行某些动作（调用工具）
    if "actions" in chunk:
        # 遍历所有动作
        for action in chunk["actions"]:
            # 打印调用的工具名称和输入参数
            # action.tool: 工具名称
            # action.tool_input: 传给工具的参数
            print(f"Calling Tool: `{action.tool}` with input `{action.tool_input}`")

    # 情况2：数据块包含 "steps" 键
    # 表示工具执行完成，返回了结果
    elif "steps" in chunk:
        # 遍历所有步骤
        for step in chunk["steps"]:
            # 打印工具的返回结果
            # step.observation: 工具执行后的观察结果
            print(f"Tool Result: `{step.observation}`")

    # 情况3：数据块包含 "output" 键
    # 表示智能体已经生成了最终答案
    elif "output" in chunk:
        # 打印最终输出
        print(f'Final Output: {chunk["output"]}')

    # 情况4：未知的数据块类型
    else:
        # 抛出异常，提示开发人员检查数据结构
        raise ValueError()

    # 打印分隔线
    print("---")


# ==================== 安全的日志记录器 ====================

class SafeLogger(BaseCallbackHandler):
    """
    安全的日志记录器

    继承自 BaseCallbackHandler

    作用：
    避免在回调过程中尝试序列化复杂的对象
    防止抛出 NotImplementedError 异常

    问题背景：
    某些回调处理器在尝试打印或序列化LLM返回的消息对象时
    可能会遇到不支持的操作，导致程序崩溃
    这个类提供了一个安全的处理方式
    """

    def on_llm_end(self, response, **kwargs):
        """
        当 LLM 完成时调用

        简单地打印一条消息，不进行复杂的对象操作

        参数：
            response: LLM的响应对象
            **kwargs: 其他参数
        """

        # 打印提示信息
        print("LLM finished. Skipping object serialization.")

        # 不尝试序列化 LLM 返回的消息
        # 避免抛出 NotImplementedError
        # （这里什么都不做，只是跳过）


# ==================== 创建安全的智能体执行器 ====================

# 创建一个带有安全日志记录器的智能体执行器

agent_executor_safe = AgentExecutor(
    agent=agent,  # 智能体对象
    tools=tools,  # 工具列表
    callbacks=[SafeLogger()],  # 添加安全日志记录器
    verbose=True,  # 开启详细模式
).with_config({"run_name": "Agent"})


# 这个执行器更安全，不会因为序列化问题而崩溃


# ==================== 函数2：异步流式事件处理 ====================

async def ask_2(question):
    """
    咨询智能体 - 高级异步版本

    使用 astream_events() 方法获取详细的执行事件
    可以监控智能体的每一个操作步骤

    参数：
        question (str): 用户的问题

    工作流程：
    1. 异步遍历智能体发出的所有事件
    2. 根据事件类型执行不同的处理逻辑
    3. 打印智能体启动、工具调用、token生成等信息

    优点：
    - 可以获取最细粒度的执行信息
    - 可以实时监控智能体的每一步操作
    - 适合调试和深入学习智能体工作原理

    缺点：
    - 需要使用 async/await 异步编程
    - 代码复杂度较高
    """

    # 使用异步for循环遍历智能体发出的事件
    # astream_events() 返回一个异步迭代器

    async for event in agent_executor_safe.astream_events(
            {"input": question},  # 输入数据
            version="v1",  # 事件格式版本
    ):
        # 获取事件的类型
        kind = event["event"]

        # 情况1：链开始事件
        if kind == "on_chain_start":
            # 检查是否是智能体的启动事件
            if (
                    event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                # 打印智能体启动信息
                print(
                    f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
                )
                # 例如："Starting agent: Agent with input: 猫藏在哪里？"

        # 情况2：链结束事件
        elif kind == "on_chain_end":
            # 检查是否是智能体的结束事件
            if (
                    event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                # 打印换行和分隔线
                print()
                print("--")

                # 打印智能体完成信息和最终输出
                print(
                    f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
                )
                # 例如："Done agent: Agent with output: 猫藏在床下..."

        # 情况3：聊天模型流式输出事件
        if kind == "on_chat_model_stream":
            # 从事件数据中提取token内容
            content = event["data"]["chunk"].content

            # 如果内容不为空
            if content:
                # Empty content in the context of OpenAI means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                #
                # 翻译：
                # 在OpenAI的上下文中，空内容表示模型请求调用工具
                # 所以我们只打印非空内容

                # 打印token内容，用 | 分隔
                print(content, end="|")

        # 情况4：工具开始执行事件
        elif kind == "on_tool_start":
            # 打印分隔线
            print("--")

            # 打印工具启动信息
            print(
                f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
            )
            # 例如："Starting tool: get_items with inputs: {'place': 'bed'}"

        # 情况5：工具执行完成事件
        elif kind == "on_tool_end":
            # 打印工具完成信息
            print(f"Done tool: {event['name']}")

            # 打印工具的输出结果
            print(f"Tool output was: {event['data'].get('output')}")

            # 打印分隔线
            print("--")


# ==================== 创建流式智能体执行器 ====================

# 创建一个带有自定义回调处理器的智能体执行器
# 用于测试方法2的流式输出

agent_executor_stream = AgentExecutor(
    agent=agent,  # 智能体对象
    tools=tools,  # 工具列表
    verbose=False,  # 关闭详细模式（因为我们自己处理输出）
    callbacks=[CallbackHandler()]  # 添加自定义回调处理器
).with_config(
    {"run_name": "Agent_stream"}  # 设置执行器名称
)


# 注意：下面这行代码被注释掉了
# agent_executor_stream.agent.llm.callbacks =[CallbackHandler()]
# 原因：已经在创建AgentExecutor时通过callbacks参数设置了回调
# 不需要再单独设置llm的回调


# ==================== 函数3：咨询智能体（流式版本） ====================

def ask_3(question):
    """
    咨询智能体 - 流式输出版本

    使用 invoke() 方法执行智能体
    流式输出由 CallbackHandler 自动处理

    参数：
        question (str): 用户的问题

    工作流程：
    1. 调用 agent_executor_stream.invoke() 方法
    2. CallbackHandler 会自动捕获每个token
    3. 在 on_llm_new_token 方法中实时打印token

    注意：
    虽然使用了 invoke()（同步调用），但由于配置了 CallbackHandler
    仍然可以实现流式输出效果
    """

    # 调用智能体执行器的 invoke 方法
    # 这会阻塞直到智能体完成所有工作
    # 但在这期间，CallbackHandler 会实时打印token

    agent_executor_stream.invoke({"input": question})

    """
    以下是被注释掉的替代方案：
    
    for chunk in agent_executor_stream.stream({"input": question}):
        pass
        # print(chunk)
    
    这种方式使用 stream() 方法逐个获取数据块
    但在这里我们依赖 CallbackHandler 来处理流式输出
    所以不需要手动处理 chunk
    """


# ==================== 主程序入口 ====================

if __name__ == '__main__':
    """
    主程序入口
    当直接运行这个脚本时，会执行这里的代码
    """

    # 以下是被注释掉的测试代码

    # 测试单个工具
    # place = where_cat_is_hiding.invoke({})
    # items = get_items.invoke({"place": "shelf"})

    # 测试基础版本的 ask 函数
    # ask("what's items are located where the cat is hiding?")
    # 这个问题需要智能体：
    # 1. 先调用 where_cat_is_hiding 工具查询猫的位置
    # 2. 再调用 get_items 工具查询该位置的物品
    # 3. 综合两个结果给出最终答案

    """
    测试异步版本的 ask_2 函数
    
    import asyncio
    asyncio.run(ask_2("where is the cat hiding? what items are in that location?"))
    
    这个测试需要：
    1. 导入 asyncio 模块
    2. 使用 asyncio.run() 运行异步函数
    3. 会显示详细的执行事件
    """

    # 测试流式版本的 ask_3 函数

    # 第一个问题：英文问题
    ask_3("what's items are located where the cat is hiding?")
    # 期望行为：
    # 1. 智能体调用 where_cat_is_hiding 工具
    # 2. 获取猫的位置（例如："under the bed"）
    # 3. 智能体调用 get_items 工具，传入 place="bed"
    # 4. 获取物品列表（例如："socks, shoes and dust bunnies"）
    # 5. 综合信息，生成最终答案
    # 6. CallbackHandler 实时流式输出最终答案

    # 第二个问题：中文问题
    ask_3("请参考哪吒闹海的故事架构，写一篇200-300字的神话故事。")
    # 这是一个创意写作任务
    # 期望看到AI逐字生成故事的流式效果
    # 但由于注释说"未成功"，可能实际运行时没有看到预期的流式输出
