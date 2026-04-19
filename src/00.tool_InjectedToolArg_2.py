#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-01-08
# @function: 防止LLM生成某些参数
# @version : V0.5
# @Description ：防止LLM生成某些参数。

"""
这个脚本演示了如何使用 InjectedToolArg 来防止大语言模型（LLM）生成某些工具参数，
而是在运行时由应用程序直接注入这些参数。

核心问题：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
在某些场景下，工具需要一些只在运行时才知道的参数值，例如：
- 用户ID（user_id）
- 会话令牌（session_token）
- API密钥（api_key）
- 请求来源IP等

这些问题参数有一个共同特点：
❌ 不应该由LLM决定或生成
✓ 应该由应用程序逻辑固定

安全风险：
如果允许LLM控制用户ID，可能会导致：
1. 用户A可以访问用户B的数据
2. 恶意用户可能伪造身份
3. 数据泄露和权限绕过

解决方案：
使用 InjectedToolArg 标记某些参数，让LLM"看不见"这些参数，
然后在运行时由应用程序自动注入正确的值。

工作流程：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 定义工具时，用 InjectedToolArg 标记敏感参数
2. LLM看到工具 schema 时，看不到被标记的参数
3. LLM生成工具调用时，不会包含这些参数
4. 应用程序在调用工具前，自动注入这些参数的值
5. 工具执行时使用注入的真实值

类比理解：
就像一个餐厅的点餐系统：
- 顾客（LLM）只能选择菜品（pets）
- 服务员（应用程序）自动添加桌号（user_id）
- 厨房（工具）收到完整的订单（菜品 + 桌号）
- 顾客无法伪造桌号，确保账单准确
"""

# ==================== 导入必要的库 ====================

# https://python.langchain.com/docs/how_to/tool_runtime/
# 参考LangChain官方文档关于工具运行时参数注入的说明

# 导入深拷贝函数，用于复制对象而不共享引用
from copy import deepcopy

# 导入列表类型注解
from typing import List

# 导入 chain 装饰器，用于将函数转换为可链接的组件
from langchain_core.runnables import chain

# 导入 InjectedToolArg 和 tool 装饰器
# InjectedToolArg: 标记某个参数应该由运行时注入，而不是由LLM生成
# tool: 将普通函数转换为LangChain工具
from langchain_core.tools import InjectedToolArg, tool

# 导入Ollama聊天模型接口
from langchain_ollama import ChatOllama

# 导入Pydantic的数据模型类和字段定义
# BaseModel: 用于定义数据结构
# Field: 用于定义字段的元数据（描述、验证等）
from pydantic import BaseModel, Field

# 导入 Annotated 类型注解
# 用于给类型添加额外的元数据（如 InjectedToolArg 标记）
from typing_extensions import Annotated

# ==================== 核心概念说明 ====================

"""
您可能需要将仅在运行时才知道的值绑定到工具。例如，工具逻辑可能需要使用发出请求的用户的 ID。
大多数情况下，此类值不应由 LLM 控制。事实上，允许 LLM 控制用户 ID 可能会导致安全风险。
相反，LLM 应该只控制本应由 LLM 控制的工具参数，而其他参数（如用户 ID）应由应用程序逻辑固定。
本操作指南向您展示了如何防止模型生成某些工具参数并在运行时直接注入它们。
"""

# ==================== 创建AI模型实例 ====================

# 创建ChatOllama模型实例
llm = ChatOllama(model="llama3.1", temperature=0.1, verbose=True)
"""
参数说明：

model="llama3.1": 
  → 使用 Meta 公司的 Llama 3.1 模型

temperature=0.1: 
  → 温度值设得很低（0.1）
  → 因为这是一个工具调用场景，需要准确、一致的输出
  → 不需要创意或多样性
  → 低温度确保LLM总是生成相同的工具调用格式

verbose=True: 
  → 开启详细模式，显示调试信息

temperature 参数的深入解释：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
temperature 就像调节"创造性"的旋钮：

低温模式（0.1-0.3）：
  ✓ 优点：回答准确、一致、可预测
  ✗ 缺点：可能显得单调、缺乏创意
  适用场景：工具调用、数据提取、代码生成
  
  例如问："提取用户ID"
  每次都会返回相同格式的结果

高温模式（0.7-1.0）：
  ✓ 优点：回答多样、有创意、有趣
  ✗ 缺点：可能不准确、跑题、甚至胡说
  适用场景：写故事、头脑风暴、创意写作

为什么这里选择0.1？
  → 工具调用需要极高的准确性
  → 我们不需要LLM发挥创意
  → 需要确保每次生成的工具调用格式一致
"""

# ==================== 数据存储 ====================

# 创建一个全局字典，用于存储用户和他们最喜欢的宠物
# 键（key）：用户ID（字符串）
# 值（value）：该用户最喜欢的宠物列表（字符串列表）

user_to_pets = {}


# 例如：
# {
#   "123": ["cats", "parrots"],
#   "456": ["dogs", "fish"]
# }
#
# 这个字典模拟了一个数据库或缓存
# 在实际应用中，可能会使用真正的数据库


# ==================== 方法1：使用 Pydantic Schema 定义工具 ====================

# Hiding arguments from the model（对模型隐藏参数）


class UpdateFavoritePetsSchema(BaseModel):
    """
    更新最喜爱宠物列表的工具参数schema

    这个类定义了 update_favorite_pets 工具的输入参数结构

    继承自 BaseModel：
    → Pydantic 的数据模型基类
    → 提供数据验证、序列化等功能
    """

    # 定义 pets 字段：宠物列表
    pets: List[str] = Field(
        ...,  # ... 表示这是必填字段
        description="最喜爱的宠物列表。"  # 字段描述，LLM会看到这个说明
    )
    # 类型：List[str] 表示字符串列表
    # 例如：["cats", "dogs", "birds"]
    #
    # 这个字段对LLM可见
    # LLM会根据用户的问题生成这个列表

    # 定义 user_id 字段：用户ID
    user_id: Annotated[str, InjectedToolArg] = Field(
        ...,  # ... 表示这是必填字段
        description="用户ID。"  # 字段描述
    )
    # 类型：Annotated[str, InjectedToolArg]
    # → str: 基本类型是字符串
    # → InjectedToolArg: 特殊标记，表示这个参数应该由运行时注入
    #
    # 关键区别：
    # → 这个字段对LLM不可见！
    # → LLM在生成工具调用时，不会包含 user_id
    # → 应用程序会在调用工具前自动注入 user_id 的值


@tool(args_schema=UpdateFavoritePetsSchema)
def update_favorite_pets(pets, user_id):
    """
    添加或者更新最喜爱的宠物列表

    这个工具用于保存用户最喜欢的宠物信息

    参数：
        pets (List[str]): 宠物列表，由LLM生成
            例如：["cats", "parrots"]

        user_id (str): 用户ID，由运行时注入
            例如："123"
            LLM看不到这个参数！

    工作原理：
    1. 接收宠物列表和用户ID
    2. 将宠物列表保存到 user_to_pets 字典中
    3. 如果该用户已有记录，则覆盖旧记录

    返回值：
        None（无返回值，只是保存数据）

    举例说明：
    调用 update_favorite_pets(["cats", "dogs"], "123")
    → user_to_pets["123"] = ["cats", "dogs"]
    """

    # 将宠物列表保存到字典中
    # 以 user_id 为键，pets 为值
    user_to_pets[user_id] = pets
    # 例如：
    # user_id = "123"
    # pets = ["cats", "parrots"]
    # 执行后：user_to_pets = {"123": ["cats", "parrots"]}


# ==================== 方法2：使用文档字符串定义工具 ====================

@tool
def delete_favorite_pets(user_id: Annotated[str, InjectedToolArg]) -> None:
    """
    删除最喜爱的宠物列表

    Delete the list of favorite pets.

    这个工具用于删除用户保存的宠物列表

    参数：
        user_id (str): 用户ID，由运行时注入
            例如："123"
            LLM看不到这个参数！

    返回值：
        None（无返回值，只是删除数据）

    工作原理：
    1. 检查 user_id 是否存在于 user_to_pets 字典中
    2. 如果存在，删除该用户的记录
    3. 如果不存在，什么都不做

    注意：
    这个函数只有一个参数 user_id，而且被标记为 InjectedToolArg
    → 意味着LLM完全看不到任何参数
    → LLM只需要决定"是否调用这个工具"
    → 具体的 user_id 由应用程序注入

    举例说明：
    用户说："删除我的宠物列表"
    → LLM决定调用 delete_favorite_pets 工具
    → LLM生成的调用：{"name": "delete_favorite_pets", "args": {}}
    → 注意 args 是空的！
    → 应用程序注入 user_id："123"
    → 实际调用：delete_favorite_pets(user_id="123")
    """

    # 打印调试信息，显示工具被调用
    print(f'delete_favorite_pets is called:{user_id}')

    # 检查用户ID是否在字典中
    if user_id in user_to_pets:
        # 如果存在，删除该用户的记录
        del user_to_pets[user_id]
        # 例如：
        # 删除前：user_to_pets = {"123": ["cats", "parrots"]}
        # 删除后：user_to_pets = {}


@tool
def list_favorite_pets(user_id: Annotated[str, InjectedToolArg]) -> None:
    """
    列出最喜爱的宠物

    List favorite pets if any.

    这个工具用于查询用户保存的宠物列表

    参数：
        user_id (str): 用户ID，由运行时注入
            例如："123"
            LLM看不到这个参数！

    返回值：
        List[str]: 宠物列表，如果用户没有记录则返回空列表
            例如：["cats", "parrots"]
            或：[]

    工作原理：
    1. 从 user_to_pets 字典中查找 user_id 对应的宠物列表
    2. 如果找到，返回宠物列表
    3. 如果没找到，返回空列表 []

    举例说明：
    用户说："我最喜欢的宠物是什么？"
    → LLM决定调用 list_favorite_pets 工具
    → LLM生成的调用：{"name": "list_favorite_pets", "args": {}}
    → 应用程序注入 user_id："123"
    → 实际调用：list_favorite_pets(user_id="123")
    → 返回：["cats", "parrots"]
    """

    # 打印调试信息，显示工具被调用
    print(f'list_favorite_pets is called:{user_id}')

    # 从字典中获取用户的宠物列表
    # .get() 方法：如果键不存在，返回默认值 []
    return user_to_pets.get(user_id, [])
    # 例如：
    # user_to_pets = {"123": ["cats", "parrots"]}
    # user_id = "123"
    # 返回：["cats", "parrots"]
    #
    # 如果 user_id = "999"（不存在）
    # 返回：[]


# ==================== 验证 InjectedToolArg 的效果 ====================

# If we look at the input schemas for these tools, we'll see that user_id is still listed:
# 如果我们查看工具的输入schema，会发现 user_id 仍然列在其中

print(f'get_input_schema:{update_favorite_pets.get_input_schema().model_json_schema()}')
# 输出示例：
# {
#   "properties": {
#     "pets": {"description": "最喜爱的宠物列表。", "items": {"type": "string"}, "title": "Pets", "type": "array"},
#     "user_id": {"description": "用户ID。", "title": "User Id", "type": "string"}
#   },
#   "required": ["pets", "user_id"],
#   "title": "UpdateFavoritePetsSchema",
#   "type": "object"
# }
#
# 注意：user_id 在这里仍然可见
# 因为 get_input_schema() 返回的是工具的完整输入schema
# 包括所有参数（无论是否被标记为 InjectedToolArg）


# But if we look at the tool call schema, which is what is passed to the model for tool-calling, user_id has been removed:
# 但是如果我们查看工具调用schema（这是传递给模型用于工具调用的schema），user_id 已经被移除了

print(f'tool_call_schema:{update_favorite_pets.tool_call_schema.model_json_schema()}')
# 输出示例：
# {
#   "properties": {
#     "pets": {"description": "最喜爱的宠物列表。", "items": {"type": "string"}, "title": "Pets", "type": "array"}
#   },
#   "required": ["pets"],
#   "title": "update_favorite_pets",
#   "type": "object"
# }
#
# 注意：user_id 不见了！
# tool_call_schema 是专门用于LLM工具调用的schema
# InjectedToolArg 标记的参数会被自动移除
# LLM只会看到 pets 参数


# ==================== 测试工具的直接调用 ====================

# So when we invoke our tool, we need to pass in user_id:
# 所以当我们手动调用工具时，仍然需要传入 user_id

user_id = "123"
# 设置测试用的用户ID

# 直接调用 update_favorite_pets 工具
# 需要手动传入 pets 和 user_id 两个参数
update_favorite_pets.invoke({"pets": ["lizard", "dog"], "user_id": user_id})
# 执行过程：
# 1. 接收参数：pets=["lizard", "dog"], user_id="123"
# 2. 执行函数体：user_to_pets["123"] = ["lizard", "dog"]
# 3. 保存成功

# 打印当前的 user_to_pets 字典
print(f'user_to_pets:{user_to_pets}')
# 输出：{'123': ['lizard', 'dog']}

# 调用 list_favorite_pets 工具查询用户"123"的宠物列表
print(f'list_favorite_pets.invoke:{list_favorite_pets.invoke({"user_id": user_id})}')
# 执行过程：
# 1. 接收参数：user_id="123"
# 2. 执行函数体：return user_to_pets.get("123", [])
# 3. 返回：["lizard", "dog"]
# 输出：['lizard', 'dog']


# ==================== 测试LLM的工具调用 ====================

# But when the model calls the tool, no user_id argument will be generated:
# 但是当模型调用工具时，不会生成 user_id 参数

# 创建工具列表
tools = [
    update_favorite_pets,  # 更新宠物列表
    delete_favorite_pets,  # 删除宠物列表
    list_favorite_pets,  # 查询宠物列表
]

# 将工具绑定到LLM
# bind_tools() 方法让LLM知道可以使用这些工具
llm_with_tools = llm.bind_tools(tools)
# 返回一个新的LLM对象，具备工具调用能力
# LLM会根据用户的问题，决定是否调用工具、调用哪个工具、传入什么参数


# 定义用户的问题
query = "my favorite animals are cats and parrots"
# 翻译："我最喜欢的动物是猫和鹦鹉"
#
# 期望行为：
# 1. LLM识别出用户在表达喜欢的动物
# 2. LLM决定调用 update_favorite_pets 工具
# 3. LLM生成参数：pets=["cats", "parrots"]
# 4. LLM不会生成 user_id（因为被 InjectedToolArg 标记了）


# 将文本转化为json结构
print("---1、调用LLM，将请求转化为json结构---")

# 调用LLM，让它分析问题并决定是否需要调用工具
ai_msg = llm_with_tools.invoke(query)
# 执行过程：
# 1. 将 query 发送给 LLM
# 2. LLM分析："用户在说喜欢的动物，应该调用 update_favorite_pets"
# 3. LLM生成工具调用：
#    {
#      "name": "update_favorite_pets",
#      "args": {"pets": ["cats", "parrots"]}
#    }
# 4. 注意：args 中没有 user_id！

# 打印LLM生成的工具调用
print(f'result:{ai_msg.tool_calls}')


# 输出示例：
# [{'name': 'update_favorite_pets', 'args': {'pets': ['cats', 'parrots']}, 'id': '...', 'type': 'tool_call'}]
#
# 关键点：
# ✓ 有 "pets": ["cats", "parrots"]
# ✗ 没有 "user_id"
# 这正是我们想要的效果！


# ==================== 运行时注入参数 ====================

# Injecting arguments at runtime（在运行时注入参数）


@chain
def inject_user_id(ai_msg):
    """
    注入用户ID到工具调用中

    这个函数是一个自定义的链式组件
    它接收LLM生成的工具调用消息，并为每个工具调用注入 user_id

    参数：
        ai_msg: LLM生成的AI消息对象
            包含 tool_calls 列表

    返回值：
        List[dict]: 注入了 user_id 的工具调用列表

    工作流程：
    1. 遍历 ai_msg 中的所有工具调用
    2. 对每个工具调用进行深拷贝（避免修改原始对象）
    3. 在 args 中添加 user_id
    4. 返回修改后的工具调用列表

    为什么要深拷贝？
    → 避免修改原始的 ai_msg 对象
    → 保持数据的不可变性，避免副作用
    """

    # 创建一个空列表，用于存储修改后的工具调用
    tool_calls = []

    # 遍历AI消息中的所有工具调用
    for tool_call in ai_msg.tool_calls:
        # 对当前工具调用进行深拷贝
        # deepcopy 会创建一个完全独立的副本
        tool_call_copy = deepcopy(tool_call)

        # 在拷贝的工具调用的 args 中添加 user_id
        tool_call_copy["args"]["user_id"] = user_id
        # 例如：
        # 修改前：{"name": "update_favorite_pets", "args": {"pets": ["cats", "parrots"]}}
        # 修改后：{"name": "update_favorite_pets", "args": {"pets": ["cats", "parrots"], "user_id": "123"}}

        # 将修改后的工具调用添加到列表中
        tool_calls.append(tool_call_copy)

    # 返回注入了 user_id 的工具调用列表
    return tool_calls


# 测试 inject_user_id 函数
new_args = inject_user_id.invoke(ai_msg)
print(f'inject_user_id:{new_args}')
# 输出示例：
# [{'name': 'update_favorite_pets', 'args': {'pets': ['cats', 'parrots'], 'user_id': '123'}, 'id': '...', 'type': 'tool_call'}]
#
# 注意：现在 args 中包含了 "user_id": "123"！


# ==================== 创建工具路由器 ====================

# And now we can chain together our model, injection code, and the actual tools to create a tool-executing chain:
# 现在我们可以将模型、注入代码和实际工具链接起来，创建一个执行工具的链

# 创建工具映射表
# 将工具名称映射到工具对象
tool_map = {tool.name: tool for tool in tools}


# 例如：
# {
#   "update_favorite_pets": update_favorite_pets,
#   "delete_favorite_pets": delete_favorite_pets,
#   "list_favorite_pets": list_favorite_pets
# }
#
# 这个映射表用于根据工具名称快速找到对应的工具对象


@chain
def tool_router(tool_call):
    """
    工具路由器

    根据工具调用中的名称，返回对应的工具对象

    参数：
        tool_call (dict): 工具调用字典
            包含 "name" 键，表示工具名称

    返回值：
        Tool: 对应的工具对象

    工作原理：
    1. 从 tool_call 中提取工具名称
    2. 从 tool_map 中查找对应的工具对象
    3. 返回工具对象

    举例说明：
    输入：{"name": "update_favorite_pets", "args": {...}}
    → 提取 name = "update_favorite_pets"
    → 查找 tool_map["update_favorite_pets"]
    → 返回 update_favorite_pets 工具对象
    """

    # 根据工具名称从映射表中获取工具对象
    return tool_map[tool_call["name"]]


# ==================== 创建完整的工具执行链 ====================

# And now we can chain together our model, injection code, and the actual tools to create a tool-executing chain:
# 现在我们可以将模型、注入代码和实际工具链接起来，创建一个执行工具的链

# 使用 | 操作符将多个组件链接成一个完整的处理链
chain = llm_with_tools | inject_user_id | tool_router.map()
# 分解说明：
# 1. llm_with_tools: LLM + 工具绑定
#    → 接收用户问题，生成工具调用
#
# 2. |: 管道操作符
#
# 3. inject_user_id: 参数注入函数
#    → 接收LLM的工具调用，注入 user_id
#
# 4. |: 管道操作符
#
# 5. tool_router.map(): 工具路由器（映射版本）
#    → .map() 表示对列表中的每个元素应用 router
#    → 接收工具调用列表，返回工具执行结果列表
#
# 完整流程：
# 用户问题 → LLM生成工具调用 → 注入user_id → 路由到对应工具 → 执行工具 → 返回结果


# 调用chain
print("--通过chain实现：2、注入新参数user_id；3、直接调用tool生成结果；4、调用LLM，生成流畅的答案---")

# 调用完整的链
result = chain.invoke(query)
print(f'chain.invoke:{result}')
# 执行过程：
# 1. llm_with_tools.invoke(query)
#    → LLM分析："my favorite animals are cats and parrots"
#    → 生成工具调用：[{"name": "update_favorite_pets", "args": {"pets": ["cats", "parrots"]}}]
#
# 2. inject_user_id.invoke(...)
#    → 注入 user_id："123"
#    → 返回：[{"name": "update_favorite_pets", "args": {"pets": ["cats", "parrots"], "user_id": "123"}}]
#
# 3. tool_router.map().invoke(...)
#    → 对每个工具调用，找到对应的工具对象
#    → 调用工具：update_favorite_pets(pets=["cats", "parrots"], user_id="123")
#    → 工具执行：user_to_pets["123"] = ["cats", "parrots"]
#    → 返回工具执行结果
#
# 最终 result 包含工具的执行结果

# 打印最终的 user_to_pets 字典
print(f'now user_to_pets :{user_to_pets}')
# 输出：{'123': ['cats', 'parrots']}
#
# 验证：
# ✓ 用户"123"的宠物列表已成功更新为 ["cats", "parrots"]
# ✓ user_id 是由应用程序注入的，不是LLM生成的
# ✓ 整个过程安全可靠
