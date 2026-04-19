#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-02-04
# @function: 用langgraph实现的chatbot
# @version : V0.5

"""
这个脚本演示了如何使用 LangGraph 构建一个具有记忆功能的聊天机器人。

核心概念：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 无记忆的对话：每次对话都是独立的，AI不记得之前说过什么
2. 有记忆的对话：手动将历史消息传给AI，让它知道上下文
3. 持久化记忆：使用 LangGraph 的 MemorySaver 自动保存和恢复对话历史

什么是 LangGraph？
→ LangChain 的一个库，用于构建有状态的、多参与者的应用程序
→ 特别适合需要维护状态的应用，如聊天机器人、工作流等
→ 提供了图形化的方式来组织和管理应用流程

为什么需要记忆？
→ 没有记忆：用户说"我叫Bob"，然后问"我叫什么"，AI回答"我不知道"
→ 有记忆：AI能记住用户之前说过的话，提供连贯的对话体验

工作流程对比：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

无记忆（chat函数）：
用户："我叫Bob" → AI："你好Bob"
用户："我叫什么" → AI："我不知道" ❌

手动记忆（chat_with_memory函数）：
应用程序手动维护：["我叫Bob", "你好Bob", "我叫什么"]
用户："我叫什么" → AI："你叫Bob" ✓

自动持久化记忆（message_persistence函数）：
LangGraph 自动保存和恢复历史
线程1："我叫Bob" → 保存
线程1："我叫什么" → 恢复历史 → AI："你叫Bob" ✓
线程2："我叫什么" → 新线程，无历史 → AI："我不知道" ✓
"""

# ==================== 导入必要的库 ====================

# 导入AI消息类型，代表AI生成的回复
from langchain_core.messages import AIMessage

# 导入人类消息类型，代表用户发送的消息
from langchain_core.messages import HumanMessage

# 导入Ollama聊天模型接口
from langchain_ollama import ChatOllama

# 导入内存检查点保存器
# MemorySaver 是 LangGraph 提供的简单内存存储
# 用于保存和恢复对话状态（聊天记录）
from langgraph.checkpoint.memory import MemorySaver

# 导入 LangGraph 的核心组件
# START: 图的起始节点标记
# MessagesState: 预定义的状态schema，专门用于管理消息列表
# StateGraph: 状态图类，用于构建有状态的工作流
from langgraph.graph import START, MessagesState, StateGraph


# ==================== 函数1：无记忆的聊天 ====================

def chat(model_name):
    """
    测试无记忆的聊天

    这个函数展示了没有记忆功能时的问题：
    AI无法记住之前的对话内容

    参数：
        model_name (str): 要使用的模型名称
            例如："llama3.1"、"deepseek-r1:14b"

    演示过程：
    1. 第一轮：用户说"我叫Bob"
    2. 第二轮：用户问"我叫什么"
    3. 结果：AI回答"我不知道"，因为它不记得第一轮对话

    这个问题说明了为什么需要记忆功能！
    """

    # 创建ChatOllama模型实例
    model = ChatOllama(model=model_name, temperature=0.3, verbose=True)

    # 第一轮对话：用户自我介绍
    response = model.invoke([HumanMessage(content="Hi! I'm Bob")])
    # 执行过程：
    # 1. 创建消息列表：[HumanMessage("Hi! I'm Bob")]
    # 2. 发送给LLM
    # 3. LLM回复："Hello Bob! How can I assist you today?"
    # 4. 返回响应对象

    # 打印AI的回复
    print(f'chat_with_no_memory:\n{response.content}')
    # 输出示例："Hello Bob! How can I assist you today?"

    # We can see that it doesn't take the previous conversation turn into context, and cannot answer the question. This makes for a terrible chatbot experience!
    # 我们可以看到，它没有考虑之前的对话上下文，无法回答问题。这使得聊天机器人体验很糟糕！

    # 第二轮对话：用户询问自己的名字
    response = model.invoke([HumanMessage(content="What's my name?")])
    # 关键问题：
    # → 这里只发送了当前问题，没有包含之前的对话历史
    # → LLM看不到第一轮对话中用户说"我叫Bob"
    # → 所以LLM无法回答这个问题

    # 打印AI的回复
    print(f'chat_with_no_memory 2:\n{response.content}')
    # 输出示例："I don't know your name." 或 "You haven't told me your name."
    # 这就是没有记忆功能的问题！


# ==================== 函数2：手动管理记忆的聊天 ====================

def chat_with_memory(model_name):
    """
    具有记忆功能的聊天（手动方式）

    这个函数展示了如何通过手动传递历史消息来实现记忆功能

    参数：
        model_name (str): 要使用的模型名称

    工作原理：
    1. 应用程序手动维护一个消息列表
    2. 每次调用LLM时，将整个消息列表传给它
    3. LLM可以看到所有历史消息，从而理解上下文

    优点：
    ✓ 简单直观
    ✓ 完全控制历史记录

    缺点：
    ✗ 需要手动管理消息列表
    ✗ 消息列表会越来越大，占用内存
    ✗ 没有持久化，程序重启后历史丢失
    ✗ 不支持多用户/多线程
    """

    # 创建ChatOllama模型实例
    model = ChatOllama(model=model_name, temperature=0.3, verbose=True)

    # 调用LLM，传入完整的对话历史
    response = model.invoke(
        [
            # 第一条消息：用户自我介绍
            HumanMessage(content="Hi! I'm Bob"),

            # 第二条消息：AI的回复
            # 注意：这里我们硬编码了AI的回复
            # 在实际应用中，这应该是上一轮LLM的真实回复
            AIMessage(content="Hello Bob! How can I assist you today?"),

            # 第三条消息：用户询问自己的名字
            HumanMessage(content="What's my name?"),
        ]
    )
    # 执行过程：
    # 1. 创建包含3条消息的列表
    # 2. 发送给LLM
    # 3. LLM看到完整的历史：
    #    - 用户说"我叫Bob"
    #    - AI说"你好Bob"
    #    - 用户问"我叫什么"
    # 4. LLM根据上下文理解，回答："你叫Bob"

    # 打印AI的回复
    print(f'chat_with_memory:\n{response.content}')
    # 输出示例："Your name is Bob!"
    # ✓ 这次AI正确回答了问题，因为它看到了完整的对话历史


# ==================== 函数3：构建带持久化记忆的LangGraph应用 ====================

# Message persistence（消息持久化）

def build_app(model_name):
    """
    构建一个带持久化记忆的LangGraph应用

    这个函数展示了如何使用 LangGraph 和 MemorySaver 实现自动的记忆管理

    参数：
        model_name (str): 要使用的模型名称

    返回值：
        CompiledStateGraph: 编译后的LangGraph应用对象
            可以多次调用，自动管理对话历史

    LangGraph 核心概念：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    1. State（状态）：
       → 应用的当前状态，包含所有需要维护的数据
       → 在这里，状态就是消息列表（MessagesState）

    2. Node（节点）：
       → 状态图中的处理单元
       → 每个节点接收当前状态，执行操作，返回状态更新
       → 在这里，只有一个节点："model"，负责调用LLM

    3. Edge（边）：
       → 连接节点的箭头，定义执行流程
       → 在这里：START → model

    4. Checkpointer（检查点保存器）：
       → 负责保存和恢复状态
       → MemorySaver 将状态保存在内存中
       → 可以使用其他保存器（如数据库）实现持久化

    5. Thread（线程）：
       → 每个 thread_id 代表一个独立的对话会话
       → 不同线程的状态互不干扰
       → 类似多个用户同时使用聊天机器人

    工作流程：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    第一次调用 app.invoke()：
    1. 检查 thread_id 对应的历史状态
    2. 如果没有，创建新状态
    3. 将新消息添加到状态中
    4. 调用 call_model 节点
    5. LLM生成回复
    6. 将回复添加到状态中
    7. 保存状态到 MemorySaver
    8. 返回完整状态

    第二次调用 app.invoke()（相同 thread_id）：
    1. 从 MemorySaver 恢复之前的状态
    2. 将新消息添加到状态中
    3. 调用 call_model 节点（LLM看到完整历史）
    4. LLM生成回复
    5. 更新并保存状态
    6. 返回完整状态
    """

    # 创建ChatOllama模型实例
    model = ChatOllama(model=model_name, temperature=0.3, verbose=True)

    # Define the function that calls the model
    # 定义调用模型的函数

    def call_model(state: MessagesState):
        """
        调用LLM模型的节点函数

        这是LangGraph中的一个节点（Node）
        它接收当前状态，调用LLM，返回状态更新

        参数：
            state (MessagesState): 当前状态
                是一个字典，包含 "messages" 键
                "messages" 的值是消息列表

        返回值：
            dict: 状态更新
                {"messages": response} 表示将AI的回复添加到消息列表中

        工作流程：
        1. 从状态中提取消息列表：state["messages"]
        2. 调用LLM，传入所有历史消息
        3. LLM生成回复
        4. 返回 {"messages": response}
           → LangGraph会自动将这个回复追加到状态的消息列表中

        注意：
        → 节点函数不需要手动管理历史
        → LangGraph和MemorySaver会自动处理
        """

        # 调用LLM，传入状态中的所有消息
        # state["messages"] 包含完整的对话历史
        response = model.invoke(state["messages"])

        # 返回状态更新
        # LangGraph会将这个回复追加到消息列表中
        return {"messages": response}

    # Define a new graph
    # 定义一个新的状态图

    # 创建状态图实例
    # state_schema=MessagesState 指定状态的schema
    # MessagesState 是LangGraph预定义的，专门用于管理消息列表
    workflow = StateGraph(state_schema=MessagesState)

    # Define the (single) node in the graph
    # 定义图中的（单个）节点

    # 添加起始边：从 START 到 "model" 节点
    # 这意味着图从 "model" 节点开始执行
    workflow.add_edge(START, "model")

    # 添加 "model" 节点
    # 节点名称："model"
    # 节点函数：call_model
    # 当图执行到 "model" 节点时，会调用 call_model 函数
    workflow.add_node("model", call_model)

    # Add memory
    # 添加记忆功能

    # 创建内存检查点保存器
    # MemorySaver 将状态保存在内存中
    # 程序重启后，记忆会丢失
    # 如果需要持久化，可以使用其他保存器（如 SQLite、PostgreSQL）
    memory = MemorySaver()

    # 编译工作流，生成可执行的应用
    # checkpointer=memory 指定使用 MemorySaver 来保存和恢复状态
    app = workflow.compile(checkpointer=memory)

    # 返回编译后的应用
    return app
    # 返回的 app 对象可以多次调用
    # 每次调用都会自动管理对话历史


# ==================== 函数4：测试持久化记忆 ====================

def message_persistence(model_name):
    """
    测试消息持久化功能

    这个函数展示了 LangGraph 如何自动管理对话历史
    支持多个独立的对话线程（类似多用户）

    参数：
        model_name (str): 要使用的模型名称

    测试场景：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    场景1：同一线程的多轮对话
    线程 "abc123"：
    - 第1轮："Hi! I'm Bob." → AI记住
    - 第2轮："What's my name?" → AI回答"Bob" ✓

    场景2：不同线程的隔离
    线程 "abc234"：
    - 第1轮："What's my name?" → AI不知道（新线程，无历史）✓

    场景3：切换回原线程
    线程 "abc123"：
    - 第3轮："What's my name?" → AI仍然记得"Bob" ✓

    这证明了：
    ✓ 同一线程内，历史被保留
    ✓ 不同线程之间，历史相互隔离
    ✓ 可以随时切换线程，历史不会混淆
    """

    # 构建LangGraph应用
    app = build_app(model_name)
    # 返回的应用对象包含了：
    # - LLM模型
    # - 状态图（call_model节点）
    # - MemorySaver（记忆保存器）

    # 配置线程ID
    # thread_id 用于标识一个独立的对话会话
    # 不同的 thread_id 对应不同的对话历史
    config = {"configurable": {"thread_id": "abc123"}}
    # 配置结构说明：
    # {
    #   "configurable": {          # 可配置的选项
    #     "thread_id": "abc123"    # 线程ID
    #   }
    # }
    #
    # 这个 thread_id 会被传递给 MemorySaver
    # 用来找到对应的对话历史

    # 第一轮对话：用户自我介绍
    query = "Hi! I'm Bob."
    # 翻译："嗨！我是Bob。"

    # 创建输入消息列表
    input_messages = [HumanMessage(query)]

    # 调用应用，执行对话
    output = app.invoke({"messages": input_messages}, config)
    # 执行过程：
    # 1. 检查 thread_id="abc123" 是否有历史状态
    #    → 第一次调用，没有历史
    # 2. 创建新状态，添加输入消息
    # 3. 执行 call_model 节点
    #    → LLM看到：[HumanMessage("Hi! I'm Bob.")]
    #    → LLM回复："Hello Bob! Nice to meet you!"
    # 4. 将回复添加到状态中
    # 5. 保存状态到 MemorySaver（关联 thread_id="abc123"）
    # 6. 返回完整状态

    # 打印AI的回复
    # output["messages"] 包含所有消息（输入 + 输出）
    # [-1] 取最后一条消息（AI的回复）
    # pretty_print() 格式化打印消息
    print(output["messages"][-1].pretty_print())  # output contains all messages in state
    # 输出示例：
    # ================================== Ai Message ==================================
    # Hello Bob! Nice to meet you! How can I help you today?

    # 第二轮对话：用户询问自己的名字
    query = "What's my name?"
    # 翻译："我叫什么名字？"

    # 创建输入消息列表
    input_messages = [HumanMessage(query)]

    # 调用应用，执行对话（使用相同的 thread_id）
    output = app.invoke({"messages": input_messages}, config)
    # 执行过程：
    # 1. 检查 thread_id="abc123" 是否有历史状态
    #    → 有！从 MemorySaver 恢复之前的状态
    #    → 历史状态：[HumanMessage("Hi! I'm Bob."), AIMessage("Hello Bob...")]
    # 2. 添加新的输入消息
    #    → 现在状态中有3条消息
    # 3. 执行 call_model 节点
    #    → LLM看到完整历史：
    #      [
    #        HumanMessage("Hi! I'm Bob."),
    #        AIMessage("Hello Bob..."),
    #        HumanMessage("What's my name?")
    #      ]
    #    → LLM根据上下文回答："Your name is Bob!"
    # 4. 将回复添加到状态中
    # 5. 更新 MemorySaver 中的状态
    # 6. 返回完整状态

    # 打印AI的回复
    print(output["messages"][-1].pretty_print())
    # 输出示例：
    # ================================== Ai Message ==================================
    # Your name is Bob!

    # different thread_id
    # 使用不同的线程ID

    # 切换到新的线程
    config = {"configurable": {"thread_id": "abc234"}}
    # 这是一个全新的线程，没有任何历史

    # 在新线程中询问名字
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    # 执行过程：
    # 1. 检查 thread_id="abc234" 是否有历史状态
    #    → 没有！这是一个新线程
    # 2. 创建新状态，只包含当前输入消息
    # 3. 执行 call_model 节点
    #    → LLM看到：[HumanMessage("What's my name?")]
    #    → LLM没有上下文，无法知道用户的名字
    #    → LLM回复："I don't know your name. You haven't told me."
    # 4. 保存新状态到 MemorySaver（关联 thread_id="abc234"）
    # 5. 返回完整状态

    # 打印AI的回复
    print(output["messages"][-1].pretty_print())
    # 输出示例：
    # ================================== Ai Message ==================================
    # I don't know your name. You haven't told me yet.
    #
    # ✓ 这证明了不同线程的历史是隔离的！

    # 切换回原来的线程
    config = {"configurable": {"thread_id": "abc123"}}
    # 回到之前的线程，应该有历史

    # 再次询问名字
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    # 执行过程：
    # 1. 检查 thread_id="abc123" 是否有历史状态
    #    → 有！从 MemorySaver 恢复状态
    #    → 历史状态包含之前的3条消息
    # 2. 添加新的输入消息
    #    → 现在状态中有4条消息
    # 3. 执行 call_model 节点
    #    → LLM看到完整历史
    #    → LLM回答："Your name is Bob!"
    # 4. 更新并保存状态
    # 5. 返回完整状态

    # 打印AI的回复
    print(output["messages"][-1].pretty_print())
    # 输出示例：
    # ================================== Ai Message ==================================
    # Your name is Bob!
    #
    # ✓ 这证明了切换线程后，历史仍然保留！


# ==================== 主程序入口 ====================

if __name__ == '__main__':
    """
    主程序入口
    当直接运行这个脚本时，会执行这里的代码
    """

    # 测试第一个模型：Llama 3.1
    mode_name = "llama3.1"
    print(f'----------------------------{mode_name}---------------------------')

    # 测试1：无记忆的聊天
    chat(mode_name)
    # 预期结果：
    # - 第一轮：AI正常回复
    # - 第二轮：AI说不知道用户的名字

    # 测试2：手动管理记忆的聊天
    chat_with_memory(mode_name)
    # 预期结果：
    # - AI正确回答用户的名字是Bob

    # 测试3：持久化记忆
    message_persistence(mode_name)
    # 预期结果：
    # - 同一线程内，AI记住用户的名字
    # - 不同线程之间，历史相互隔离
    # - 切换回原线程，历史仍然保留

    # 测试第二个模型：DeepSeek R1 14B
    mode_name = "deepseek-r1:14b"
    print(f'----------------------------{mode_name}---------------------------')

    # 对 DeepSeek 模型进行同样的测试
    chat(mode_name)
    chat_with_memory(mode_name)
    message_persistence(mode_name)
    # 这样可以比较不同模型的表现
