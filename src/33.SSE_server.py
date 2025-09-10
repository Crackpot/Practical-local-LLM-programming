#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @date    : 2025-04-28
# @description: 演示SSE的使用

llm_model_name = "qwen3"

import os
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

"""
1. 构建提示词
"""
def build_prompt(language_dst:str,text:str):
    """构建提示词
    在不指定源语言的情况下，LLM也可以翻译。
    """
    prompt_sysytem = "你是一个专业的翻译助手。"
    prompt_user = "你是一个专业的翻译助手。请将以下文本翻译成客户指定语言。只输出翻译结果，不要包含任何其他解释、说明或额外文本。默认翻译成中英文互译，用户指定语言的话使用指定语言。"
    prompt_user = f"""你是一个专业的翻译助手。请将以下文本翻译成 {language_dst}。只输出翻译结果，不要包含任何其他解释、说明或额外文本。
        原文：
        {text}
        翻译："""
    return [
        SystemMessage(content =prompt_sysytem),
        HumanMessage(content=prompt_user)
    ]

"""
2. 大模型流式响应
"""
model = ChatOllama(model=llm_model_name,temperature=0.2,verbose=True)

def stream_generator(language_dst:str,text:str):
    """流式输出大模型的回答"""
    prompt = build_prompt(language_dst=language_dst,text=text)
    inside_think = False # 标记是否在<think>区间
    for chunk in model.stream(prompt):
        if isinstance(chunk, AIMessage):
            # 过滤掉<think>...</think>部分            
            if "<think>" in chunk.content:
                inside_think = True
            elif "</think>" in chunk.content:
                inside_think = False
                continue    # 跳过 </think>

            if not inside_think:
                yield {
                    "event": "message",
                    "data": chunk.content
                }
    yield {
        "event": "end",
        "data": "[[END]]"
    }

"""
3. 定义接口
"""

from pydantic import BaseModel,Field

class TranslateRequest(BaseModel):
    """请求消息体"""
    text: str = Field(..., min_length=1, description="要翻译的文本")
    language_dst: str = Field(..., min_length=1, description="目标语言")

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="翻译接口")

@app.post("/translate_stream",tags=["接口"],summary="翻译接口，流式返回内容")
async def stream_translation(req: TranslateRequest):
    return EventSourceResponse(stream_generator(language_dst=req.language_dst,text=req.text))

@app.get("/translate",tags=["测试客户端"],summary="返回翻译的前端界面")
async def get_translate_html():
    """返回翻译页面
    """

    file_path = os.path.join(os.path.dirname(__file__), "./33.SSE_client.html")
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

import uvicorn

if __name__ == '__main__':
    """交互式API文档地址：
    http://127.0.0.1:9000/docs/
    http://127.0.0.1:9000/redoc/
    """
   
    uvicorn.run(app, host="127.0.0.1", port=9000)
