#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-06
# @function: 扩展的聊天历史记录类。
# @version : V0.5
# @Description ：可以限制聊天记录的最大长度。max_size:设置为偶数。因为User和AI的消息会分别记录为1条，设置为偶数后，User和AI才会成对。

"""
在 https://python.langchain.com/v0.2/docs/tutorials/chatbot/ 中有使用trim_messages对消息历史进行裁剪的例子
但是这里依然需要用大模型来计算token，通过计算结果进行裁剪，比较耗费资源。

本例的方法没那么智能，也可能有时候会突破token大小限制出错，但是我想已经能解决绝大部分问题了。
"""

# 从langchain_community包导入基础的聊天历史记录类
# ChatMessageHistory 是一个用来存储和管理对话历史的工具
from langchain_community.chat_message_histories import ChatMessageHistory

# 导入基础消息类型，所有具体的消息类型（如用户消息、AI消息）都继承自这个基类
from langchain_core.messages import BaseMessage


class MessageHistory(ChatMessageHistory):
    """
    扩展的聊天历史记录类。可以限制聊天记录的最大长度。

    工作原理：
    想象你在和一个AI助手聊天，每次你说一句话，AI回复一句话，都会被记录下来。
    如果不加限制，聊天记录会越来越长，占用越来越多内存，也会让AI处理变慢。
    这个类就像是一个有容量限制的记事本，当记录的消息数量超过限制时，
    会自动删除最早的消息，只保留最新的几条。

    为什么要设置偶数？
    因为一次完整的对话包含：用户说一句话 + AI回复一句话 = 2条消息
    如果设置为偶数（比如4），就能保证保存的是完整的对话轮次（2轮完整对话）
    如果设置为奇数（比如3），可能会出现只有用户的话，没有AI回复的情况

    Args:
        max_size: 最大消息数量，建议设置为偶数。因为User和AI的消息会分别记录为1条，设置为偶数后，User和AI才会成对。
    """

    def __init__(self, max_size: int):
        """
        初始化方法：创建一个新聊天记录管理器

        参数：
            max_size (int): 最多保留多少条消息。例如设置为4，就只保留最新的4条消息
        """
        # 调用父类 ChatMessageHistory 的初始化方法，获得基础的聊天记录功能
        super().__init__()

        # 保存最大消息数量限制
        self._max_size = max_size

    def add_message(self, message: BaseMessage):
        """
        添加一条新消息到聊天记录中

        这个方法会在以下情况被自动调用：
        - 用户发送了一条新消息
        - AI生成了一条回复

        参数：
            message (BaseMessage): 要添加的消息对象，可以是用户消息或AI消息

        工作流程：
        1. 先把新消息添加到聊天记录末尾
        2. 检查总消息数是否超过了限制
        3. 如果超了，就删除最早的消息，只保留最新的 max_size 条
        """
        # 首先调用父类的方法，把新消息添加到聊天记录中
        super().add_message(message)

        # 检查当前聊天记录中的消息总数是否超过了设定的最大值
        if len(self.messages) > self._max_size:
            # 如果超过了，打印提示信息
            print('消息超限，马上压缩！')

            # 关键操作：只保留最新的 max_size 条消息
            # self.messages[-self._max_size:] 是Python的切片语法
            # 负号表示从末尾开始计数，例如 [-4:] 表示取最后4条消息
            # 这样就能自动丢弃最早的旧消息
            self.messages = self.messages[-self._max_size:]


# 下面这些导入用于 SessionHistory 类
# BaseChatMessageHistory 是一个接口类型，定义了聊天记录类应该有哪些方法
from langchain_core.chat_history import BaseChatMessageHistory

# AIMessage 代表AI生成的消息，用来判断一条消息是谁发的
from langchain_core.messages import AIMessage


class SessionHistory(object):
    """
    会话历史管理器

    作用：
    想象一个客服系统，同时有多个用户在和AI聊天。每个用户的聊天记录需要分开保存，
    不能混在一起。这个类就像一个档案管理员，为每个用户（session_id）创建一个独立的
    聊天记录本（MessageHistory），并负责管理所有这些记录本。

    实际应用场景：
    - 网站上有多个用户同时使用AI助手
    - 每个用户有自己的对话历史
    - 需要根据用户ID找到对应的聊天记录

    核心概念：
    - session_id: 会话ID，可以理解为用户的唯一标识符（比如用户A、用户B）
    - store: 仓库，用来存放所有用户的聊天记录
    """

    def __init__(self, max_size: int):
        """
        初始化会话历史管理器

        参数：
            max_size (int): 每个用户的聊天记录最多保留多少条消息
        """
        # 调用父类的初始化方法（虽然object父类没什么特殊操作）
        super().__init__()

        # 保存每个聊天记录的最大消息数量限制
        self._max_size = max_size

        # 创建一个空字典，用来存储所有用户的聊天记录
        # 字典的键(key)是 session_id（用户ID）
        # 字典的值(value)是该用户对应的 MessageHistory 对象
        # 初始状态是空的 {}，随着用户增多会逐渐添加
        self._store = {}

    def process(self, session_id: str) -> BaseChatMessageHistory:
        """
        获取或创建某个用户的聊天记录

        这是这个类最重要的方法！

        工作流程：
        1. 检查这个用户（session_id）之前是否有过聊天记录
        2. 如果没有，就为他创建一个新聊天记录本（MessageHistory）
        3. 如果有，就直接返回已有的聊天记录
        4. 返回的这个对象可以用来添加新消息或查看历史

        参数：
            session_id (str): 用户的唯一标识符，比如 "user_001"、"user_002"

        返回值：
            BaseChatMessageHistory: 该用户的聊天记录对象，可以用来添加或查看消息

        举例说明：
        假设用户 "张三" 第一次来聊天：
        - process("张三") 发现 _store 中没有 "张三"
        - 创建一个新的 MessageHistory(max_size=4)
        - 存入 _store["张三"] = 新的MessageHistory对象
        - 返回这个对象

        第二次 "张三" 再来：
        - process("张三") 发现 _store 中已有 "张三"
        - 直接返回之前创建的那个MessageHistory对象
        - 这样就能看到之前的聊天记录了
        """
        # 检查这个 session_id 是否已经在仓库中存在
        if session_id not in self._store:
            # 如果不存在，说明这是这个用户第一次聊天
            # 为他创建一个新的 MessageHistory 对象，设置最大消息数量限制
            # 然后存到仓库 _store 中，键是 session_id，值是新建的 MessageHistory 对象
            self._store[session_id] = MessageHistory(max_size=self._max_size)

        # 返回这个用户对应的聊天记录对象
        # 无论是刚创建的还是之前就有的，都返回它
        return self._store[session_id]

    def print_history(self, session_id):
        """
        打印某个用户的完整聊天历史记录

        这个方法主要用于调试和查看，可以把某个用户的所有对话内容显示出来

        参数：
            session_id: 用户的唯一标识符，指定要查看哪个用户的聊天记录

        输出示例：
        显示聊天历史记录...
        User: 你好，今天天气怎么样？

        AI: 今天天气晴朗，温度适宜，适合外出活动。

        User: 谢谢！

        AI: 不客气，还有什么可以帮助你的吗？
        """
        # 打印提示文字
        print("显示聊天历史记录...")

        # 遍历这个用户的所有消息
        # self._store[session_id].messages 获取该用户的所有消息列表
        for message in self._store[session_id].messages:
            # 判断这条消息是AI发的还是用户发的
            # isinstance() 函数用来检查一个对象是否属于某个类型
            if isinstance(message, AIMessage):
                # 如果是AIMessage类型，说明是AI说的话
                prefix = "AI"
            else:
                # 否则就是用户说的话（HumanMessage类型）
                prefix = "User"

            # 打印这条消息，格式为 "AI: 内容" 或 "User: 内容"
            # message.content 是消息的实际文本内容
            # \n 是换行符，让每条消息之间空一行，更易读
            print(f"{prefix}: {message.content}\n")
