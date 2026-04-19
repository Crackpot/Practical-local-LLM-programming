#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @time    : 2025-02-08
# @function: 扩展的JsonOutputParser类。
# @version : V0.5
# @Description ：扩展的JsonOutputParser类，用来支持langchain支持的不好的模型。

"""
背景说明：
有些AI模型（比如DeepSeek）在输出JSON格式数据时，会额外添加一些"思考过程"的内容。
这些思考过程通常包裹在特殊标记中（<think>...</think>），但这会导致标准的JSON解析器出错。

举个例子，我们想要AI返回这样的JSON：
{"name": "张三", "age": 25}

但DeepSeek可能会返回：
<think>让我想想，用户想要一个包含姓名和年龄的JSON对象...</think>
{"name": "张三", "age": 25}

标准的JSON解析器看到 <think> 这些内容就会报错，因为它不是合法的JSON格式。

这个类的作用就是：
1. 检测AI的输出中是否包含这种"思考过程"
2. 如果有，就提取出其中真正的JSON部分
3. 然后再进行正常的JSON解析

就像从一个包裹中取出真正的礼物，把包装纸扔掉。
"""

# ==================== 导入必要的库 ====================

# 导入正则表达式模块
# 正则表达式是一种强大的文本匹配工具，可以用来查找、提取特定格式的文本
# 这里用来从AI的输出中提取JSON字符串
import re
# 导入类型提示
# Any 表示可以是任何类型，用于函数返回值类型标注
from typing import Any

# 从langchain_core导入标准的JSON输出解析器
# JsonOutputParser 是LangChain提供的工具，用来把AI的文本输出转换成JSON对象（Python字典）
# 我们的类将继承这个类，并在其基础上增加处理"思考过程"的功能
from langchain_core.output_parsers import JsonOutputParser
# 导入Generation类
# Generation 代表AI生成的一次输出结果，包含了AI返回的文本内容等信息
from langchain_core.outputs import Generation


# ==================== 自定义JSON解析器类 ====================

class ThinkJsonOutputParser(JsonOutputParser):
    """
    支持"思考过程"的JSON输出解析器

    继承关系：
    ThinkJsonOutputParser 继承自 JsonOutputParser
    → 拥有 JsonOutputParser 的所有功能（标准JSON解析）
    → 额外增加了处理 <think>...</think> 标记的能力

    适用场景：
    - 使用DeepSeek等会输出思考过程的AI模型
    - 需要AI返回结构化JSON数据的任务
    - 希望兼容多种不同行为的AI模型

    工作原理：
    1. 接收AI的原始输出文本
    2. 检查是否包含 <think> 和 </think> 标记
    3. 如果包含，用正则表达式提取其中的JSON部分
    4. 调用父类的解析方法，将提取出的JSON文本转换为Python对象
    """

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """
        将AI模型的输出结果解析为JSON对象（Python字典或列表）

        这是整个类的核心方法！当AI返回结果后，这个方法会被自动调用来处理输出。

        参数说明：
        -----------
        result : list[Generation]
            AI模型生成的结果列表
            - 通常只包含一个元素 result[0]
            - 每个Generation对象都有一个 .text 属性，存储AI返回的文本内容

            举例：
            result[0].text 可能是：
            '<think>我需要返回一个JSON对象...</think>\n{"name": "张三"}'

        partial : bool, 可选参数
            是否解析"部分"JSON对象
            - False（默认）：要求完整的JSON，如果JSON不完整会报错
            - True：允许解析不完整的JSON，返回已解析的部分

            举例：
            完整JSON：{"name": "张三", "age": 25}  ← partial=False时可以解析
            部分JSON：{"name": "张三",           ← partial=True时可以解析

        返回值：
        --------
        Any（通常是 dict 或 list）
            解析后的Python对象
            - JSON对象 {"key": "value"} → Python字典 {"key": "value"}
            - JSON数组 [1, 2, 3] → Python列表 [1, 2, 3]

        可能抛出的异常：
        -----------------
        OutputParserException
            当AI的输出不是有效的JSON格式时抛出
            比如：AI返回了纯文本而不是JSON

        工作流程详解：
        --------------
        步骤1：获取AI的原始输出文本
        步骤2：清理文本两端的空白字符
        步骤3：检查是否包含思考过程标记
        步骤4：如果包含，提取真正的JSON部分
        步骤5：调用父类方法进行标准JSON解析
        步骤6：返回解析后的Python对象
        """

        # ==================== 步骤1：获取AI的原始输出文本 ====================

        # 从结果列表中取出第一个（也是唯一一个）Generation对象
        # 然后获取它的 .text 属性，这就是AI返回的原始文本内容
        text = result[0].text

        # ==================== 步骤2：清理文本 ====================

        # strip() 方法去除文本开头和结尾的空白字符（空格、换行符、制表符等）
        # 这样可以让后续的处理更准确
        # 例如："  \n  {"name": "张三"}  \n  " → "{"name": "张三"}"
        text = text.strip()

        # ==================== 步骤3-4：检测并提取JSON（关键步骤）====================

        # 判断AI的输出中是否包含DeepSeek等模型的"思考过程"标记
        # <think> 和 </think> 是特殊的XML风格标记
        # <think> 表示"思考开始"
        # </think> 表示"思考结束"
        #
        # 举例：
        # 如果 text 是：
        # '<think>让我分析一下用户的需求...我觉得应该返回一个JSON对象</think>\n{"result": "ok"}'
        # 那么 '<think>' in text 为 True，'</think>' in text 也为 True
        # 条件成立，进入if代码块
        if '<think>' in text and '</think>' in text:

            # 使用正则表达式提取JSON对象
            # re.search() 在文本中搜索匹配正则表达式的第一个位置
            #
            # 正则表达式 r'\{.*}' 的含义：
            # \{     → 匹配左花括号 { （需要转义，因为{在正则中有特殊含义）
            # .*     → 匹配任意字符（除了换行符）任意次数
            # }      → 匹配右花括号 }
            #
            # re.DOTALL 标志：
            # 让 . 也能匹配换行符，这样可以跨越多行匹配JSON
            #
            # 举例：
            # text = '<think>思考...</think>\n{"name": "张三",\n"age": 25}'
            # match 将会匹配到：{"name": "张三",\n"age": 25}
            match = re.search(r'\{.*}', text.strip(), re.DOTALL)

            # 检查是否成功找到了匹配
            # 如果AI的输出中确实包含JSON对象，match就不会是None
            if match:
                # group(0) 返回整个匹配的文本
                # 将提取出的纯JSON字符串赋值给text变量
                # 现在 text 只包含JSON部分，没有了 <think>...</think> 这些干扰内容
                text = match.group(0)

        # 将处理后的文本（已经去除了思考过程，只保留JSON）写回result对象
        # 这样后续的解析就能正常工作了
        result[0].text = text

        # ==================== 步骤5-6：调用父类方法进行标准JSON解析 ====================

        # 调用父类 JsonOutputParser 的 parse_result 方法
        # 此时 result[0].text 已经是干净的JSON字符串了
        # 父类方法会：
        # 1. 验证JSON格式是否正确
        # 2. 将JSON字符串转换为Python对象（字典或列表）
        # 3. 返回转换后的对象
        #
        # 举例：
        # 输入：'{"name": "张三", "age": 25}'
        # 输出：{"name": "张三", "age": 25}  （Python字典）
        return super().parse_result(result, partial=partial)


"""
使用示例：
---------

假设我们要让AI返回一个用户信息，期望得到JSON格式：

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# 1. 创建解析器实例
parser = ThinkJsonOutputParser()

# 2. 定义提示词模板，告诉AI要返回JSON格式
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手，请以JSON格式返回用户信息，包含name和age字段"),
    ("human", "请生成用户 {username} 的信息")
])

# 3. 创建链（Chain）：提示词 → AI模型 → 解析器
chain = prompt | ChatOllama(model="deepseek-r1:14b:7b") | parser

# 4. 执行链，获取结果
result = chain.invoke({"username": "张三"})

# 5. 直接使用解析后的Python字典
print(result["name"])  # 输出：张三
print(result["age"])   # 输出：25

如果没有 ThinkJsonOutputParser，当DeepSeek返回带 <think> 的内容时会报错。
有了它，就能自动处理这种情况，让程序更加健壮！

对比说明：
---------
标准 JsonOutputParser：
  AI输出：{"name": "张三"}  ✓ 能解析
  AI输出：<think>...</think>\n{"name": "张三"}  ✗ 报错！

ThinkJsonOutputParser（本类）：
  AI输出：{"name": "张三"}  ✓ 能解析
  AI输出：<think>...</think>\n{"name": "张三"}  ✓ 也能解析！（自动提取JSON部分）
"""
