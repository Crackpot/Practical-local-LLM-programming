#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-07
# @function: AI助理
# @version : V0.5
# @Description ：修改提示词，大模型秒变助理。

"""
这个脚本实现了一个简单的AI聊天助手，具有以下特点：

核心功能：
1. 使用本地运行的大语言模型（通过Ollama）
2. 支持多轮对话，能记住之前的聊天内容
3. 可以同时服务多个用户，每个用户的对话互不干扰
4. 自动限制聊天记录长度，避免内存占用过大

工作流程：
用户提问 → 系统查找该用户的历史记录 → 将历史+新问题一起发给AI → AI生成回答 → 保存对话

类比说明：
就像一个有多个隔间的咨询室，每个来访者（用户）有自己的咨询师（AI），
咨询师会记得之前和这位来访者的对话内容，但不会混淆不同来访者的事情。
"""

# ==================== 导入必要的库 ====================

# 导入基础聊天历史接口类型
# 这是一个"标准规范"，定义了聊天记录类应该有哪些方法
from langchain_core.chat_history import BaseChatMessageHistory
# 导入人类消息类型，代表用户发送的消息
from langchain_core.messages import HumanMessage
# 导入聊天提示词模板类
# ChatPromptTemplate 用来构建发送给AI的完整指令
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 导入带有消息历史的 runnable 类
# RunnableWithMessageHistory 是一个"包装器"，它能自动处理聊天记录
# 就像一个智能文件夹，每次对话时自动附上之前的聊天记录
from langchain_core.runnables.history import RunnableWithMessageHistory
# 导入Ollama聊天模型接口，用于连接本地运行的大语言模型
# Ollama是一个可以在本地电脑上运行大模型的工具，类似一个"模型服务器"
from langchain_ollama import ChatOllama

# 导入我们自定义的会话历史管理器
# 这个类在 LimitedChatMessageHistory.py 文件中定义
# 作用：管理多个用户的聊天记录，确保每个用户的对话独立保存
from common.LimitedChatMessageHistory import SessionHistory


# ==================== 函数1：获取大语言模型 ====================

def get_llm():
    """
    创建并返回一个本地大语言模型实例

    这个函数就像是在"启动"一个AI助手，配置好它的性格和行为方式。

    返回值：
        ChatOllama对象：一个连接到本地Ollama服务的聊天模型

    关键参数说明：
    - model="llama3.1": 指定使用的模型名称
      → Llama 3.1是Meta公司开发的开源大语言模型
      → 就像选择不同的"大脑"，还有GPT、Qwen等其他选择

    - temperature=0.3: 控制AI回答的"创造力"和"随机性"
      → 温度值范围通常是0到1
      → 低温（如0.1-0.3）：回答更保守、稳定、可预测
        例如：问"1+1等于几"，总是回答"2"
      → 高温（如0.7-1.0）：回答更多样、有创意，但可能不准确
        例如：写故事时会有更多意想不到的情节
      → 这里设为0.3，因为助手需要准确回答问题，不需要太多创意

    - verbose=True: 开启详细模式
      → 会在控制台打印更多信息，方便调试
      → 可以看到AI是如何处理的

    类比理解：
    temperature就像调节收音机的音量旋钮：
    - 调低：声音清晰稳定，但可能单调
    - 调高：声音变化丰富，但可能有杂音
    """

    # 创建并返回一个ChatOllama对象
    # 这个对象代表了与本地Llama 3.1模型的连接
    return ChatOllama(model="llama3.1", temperature=0.3, verbose=True)


# ==================== 创建全局会话管理器 ====================

# 创建一个会话历史管理器实例
# max_size=20 表示每个用户最多保留20条消息（约10轮对话）
# 这个数字设置为偶数，保证用户问题和AI回答成对出现
session_history = SessionHistory(max_size=20)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    获取或创建某个用户的聊天记录

    这个函数是 RunnableWithMessageHistory 需要的"回调函数"
    每当有新的对话请求时，系统会自动调用这个函数来获取该用户的聊天记录

    参数：
        session_id (str): 用户的唯一标识符
            例如："user_001"、"liu123"等
            就像每个人的身份证号，用来区分不同的用户

    返回值：
        BaseChatMessageHistory: 该用户的聊天记录对象
            这个对象可以用来添加新消息或查看历史消息

    工作流程：
    1. 接收一个 session_id（用户ID）
    2. 调用 session_history.process() 方法
    3. process() 方法会检查这个用户是否已有聊天记录
       - 如果有：返回已有的记录
       - 如果没有：创建新的记录并返回

    举例说明：
    第一次调用 get_session_history("张三")：
    → session_history 发现没有"张三"的记录
    → 创建一个新的 MessageHistory(max_size=20)
    → 返回这个新对象

    第二次调用 get_session_history("张三")：
    → session_history 发现已有"张三"的记录
    → 直接返回之前创建的那个对象
    → 这样就能继续之前的对话了
    """

    # 调用会话管理器的 process 方法，获取该用户的聊天记录
    return session_history.process(session_id)


# ==================== 函数2：创建带历史记录的对话链 ====================

def get_history_chain():
    """
    创建并返回一个带有聊天记录功能的对话链

    什么是"对话链"（Chain）？
    → Chain就像一条流水线，把多个步骤串联起来
    → 在这个例子中：提示词模板 → AI模型 → 输出结果

    这个函数的作用：
    1. 创建一个提示词模板（告诉AI它的角色是什么）
    2. 将模板与AI模型连接起来
    3. 用 RunnableWithMessageHistory 包装，使其具有记忆功能

    返回值：
        RunnableWithMessageHistory: 一个可以自动处理历史记录的对话链对象

    类比理解：
    如果把AI对话比作做饭：
    - prompt（提示词）= 菜谱，告诉厨师要做什么菜、有什么要求
    - LLM（大模型）= 厨师，根据菜谱做菜
    - chain（对话链）= 整个厨房工作流程，从接单到出菜
    - RunnableWithMessageHistory = 智能服务员，记得每位客人的点餐历史
    """

    # 创建一个聊天提示词模板
    # ChatPromptTemplate.from_messages() 方法用来构建发送给AI的完整指令
    # 它接受一个消息列表，每个消息包含类型和内容

    prompt = ChatPromptTemplate.from_messages(
        [
            # 第一条消息：系统消息（system message）
            # 系统消息的作用是设定AI的角色和行为准则
            # 就像给演员剧本，告诉他扮演什么角色

            (
                "system",  # 消息类型：系统指令
                "You are a helpful assistant. Answer all questions to the best of your ability.",
                # 系统提示词内容（英文）：
                # "你是一个乐于助人的助手。尽你所能回答所有问题。"
                #
                # 这条指令会告诉AI：
                # 1. 你的身份：助手
                # 2. 你的态度：乐于助人
                # 3. 你的任务：尽力回答问题
                #
                # 可以修改这个提示词来改变AI的行为，例如：
                # - "你是一个专业的Python程序员，只回答编程相关问题"
                # - "你是一个幽默的聊天伙伴，用风趣的方式回答问题"
                # - "你是一个严格的老师，会指出学生的错误并给出建议"
            ),

            # 第二条消息：消息占位符（MessagesPlaceholder）
            # 这是一个"空位"，用来放置用户的聊天记录
            # variable_name="messages" 表示这个位置的变量名叫 "messages"
            # 当实际运行时，这里会被替换成真实的聊天历史

            MessagesPlaceholder(variable_name="messages"),
            # 作用说明：
            # 假设用户之前问了两个问题，AI回答了两次
            # 那么在实际发送时，"messages" 位置会被替换成：
            # [
            #   HumanMessage("问题1"),
            #   AIMessage("回答1"),
            #   HumanMessage("问题2"),
            #   AIMessage("回答2")
            # ]
            #
            # 这样AI就能看到完整的对话历史，做出连贯的回答
        ]
    )
    # 此时的 prompt 就像一个模板文件，还没有填入具体内容
    # 每次对话时，会根据实际情况填充 "messages" 部分

    # 使用 "|" 符号将提示词模板和AI模型连接起来，形成一个处理链
    # 这个符号叫做"管道操作符"（pipe operator）
    # 它的作用是把前一个组件的输出作为后一个组件的输入

    chain = prompt | get_llm()
    # 分解说明：
    # 1. prompt: 提示词模板，负责构建完整的指令
    # 2. |: 管道操作符，表示"流向"
    # 3. get_llm(): 调用前面定义的函数，获取AI模型实例
    #
    # 整个流程：
    # 用户消息 + 历史记录 → prompt 组装成完整指令 → 发送给 LLM → LLM 生成回答
    #
    # 类比理解：
    # 就像组装生产线：
    # 原料（用户输入）→ 加工机器1（prompt模板）→ 加工机器2（AI模型）→ 成品（AI回答）

    # 创建并返回一个带有消息历史功能的 runnable 对象
    # RunnableWithMessageHistory 是一个"包装器"，它给普通的对话链增加了记忆能力

    return RunnableWithMessageHistory(chain, get_session_history)
    # 参数说明：
    # 1. chain: 我们刚才创建的对话链（prompt + LLM）
    #    → 这是核心的处理能力
    #
    # 2. get_session_history: 一个回调函数（前面定义的）
    #    → 用来获取某个用户的聊天记录
    #    → 每当需要处理对话时，RunnableWithMessageHistory 会自动调用这个函数
    #
    # RunnableWithMessageHistory 的工作原理：
    # 当用户发送新消息时：
    # 1. 调用 get_session_history(session_id) 获取该用户的历史记录
    # 2. 将历史记录 + 新消息一起填入 prompt 模板
    # 3. 发送给 AI 模型进行处理
    # 4. 收到 AI 的回答后，自动将新对话保存到历史记录中
    # 5. 返回 AI 的回答
    #
    # 这样就实现了"有记忆的对话"！


# ==================== 创建全局对话链实例 ====================

# 调用 get_history_chain() 函数，创建一个带历史记录的对话链
# 这个对象会在整个程序运行期间被重复使用

with_message_history = get_history_chain()


# 为什么要放在全局？
# → 避免每次对话都重新创建，提高效率
# → 保持会话管理器的状态（保存的用户记录）
#
# 这个对象就像一个"智能客服系统"，已经配置好了：
# - AI模型（Llama 3.1）
# - 角色设定（乐于助人的助手）
# - 记忆功能（能记住每个用户的对话历史）


# ==================== 函数3：聊天接口 ====================

def chat(human_message, session_id):
    """
    与AI助手进行对话的核心函数

    这是整个脚本最重要的函数！
    其他所有代码都是为了支持这个函数的运行。

    参数：
        human_message (str): 用户发送的消息内容
            例如："你好"、"今天天气怎么样"

        session_id (str): 用户的唯一标识符
            例如："user_001"、"liu123"
            用来区分不同的用户，确保每个人的对话历史独立保存

    返回值：
        str: AI生成的回答文本内容
            例如："你好！有什么可以帮助你的吗？"

    工作流程（详细步骤）：
    1. 接收用户的消息和session_id
    2. 将消息包装成 HumanMessage 对象
    3. 调用 with_message_history.invoke() 方法
       a. 根据 session_id 查找该用户的历史记录
       b. 将历史记录 + 新消息组合成完整的提示词
       c. 发送给 AI 模型
       d. AI 生成回答
       e. 将新对话（用户消息 + AI回答）保存到历史记录
    4. 从响应对象中提取文本内容（response.content）
    5. 返回文本内容

    举例说明：
    第一次调用 chat("你好", "张三")：
    → 查找"张三"的历史记录（空的）
    → 发送："你是助手。用户说：你好"
    → AI回答："你好！有什么可以帮助你的吗？"
    → 保存这两条消息到"张三"的记录中
    → 返回："你好！有什么可以帮助你的吗？"

    第二次调用 chat("你是谁", "张三")：
    → 查找"张三"的历史记录（有之前的对话）
    → 发送："你是助手。历史：[用户:你好, AI:你好！...] 用户说：你是谁"
    → AI知道上下文，回答："我是一个AI助手..."
    → 保存新对话
    → 返回回答
    """

    # 调用对话链的 invoke() 方法，执行一次完整的对话流程
    # invoke() 是 LangChain 中标准的执行方法，意思是"调用"或"执行"

    response = with_message_history.invoke(
        # 第一个参数：要发送的消息列表
        # 这里只有一个元素，就是用户的新消息

        [HumanMessage(content=human_message)],
        # HumanMessage() 创建一个人类消息对象
        # content=human_message 设置消息的具体内容
        #
        # 为什么是列表 [] 而不是单个消息？
        # → 因为有时可能需要一次性发送多条消息
        # → 统一使用列表格式更方便
        #
        # 注意：这里只传了新消息，没有传历史记录
        # → 历史记录由 with_message_history 自动处理

        # 第二个参数：配置信息
        # config 字典用来传递各种配置选项

        config={"configurable": {"session_id": session_id}},
        # 配置结构说明：
        # {
        #   "configurable": {          # 可配置的选项
        #     "session_id": session_id # 用户ID，告诉系统这是哪个用户
        #   }
        # }
        #
        # 这个 session_id 会被传递给 get_session_history() 函数
        # 用来找到对应用户的聊天记录
        #
        # 为什么需要这么复杂的嵌套结构？
        # → LangChain 的设计允许传递多种配置
        # → "configurable" 是专门用来传递自定义配置的关键字
    )
    # invoke() 方法执行完毕后，会返回一个响应对象
    # 这个对象包含了 AI 的完整回答，包括：
    # - content: 文本内容
    # - additional_kwargs: 额外信息（如果有）
    # - 其他元数据

    # 从响应对象中提取纯文本内容并返回
    # response.content 就是 AI 生成的回答文本

    return response.content
    # 例如：如果 AI 回答 "你好！很高兴见到你。"
    # 那么 response.content 就是这个字符串
    #
    # 为什么不直接返回 response 对象？
    # → 因为调用者通常只需要文本内容
    # → 返回字符串更简单、更易用


if __name__ == '__main__':
    session_id = "liu123"

    # 测试chat方法
    print(chat("你知道x-space的老板马斯克么？", session_id))
    print(chat("他出生在哪个国家？", session_id))
    print(chat("他和特朗普是什么关系？", session_id))
    print(chat("我和特朗普是什么关系？", session_id))
    print(chat("我和马斯克是什么关系？", session_id))

    session_history.print_history(session_id)
