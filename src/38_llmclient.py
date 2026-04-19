#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @date    : 2025-10-02
# @description: 通用的大模型客户端

from openai import OpenAI


class LLMClient:
    # 在 .env 或系统环境变量里设置各家 API key（比如 OPENAI_API_KEY、ANTHROPIC_API_KEY 等）
    PROVIDERS = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY"
        },
        "anthropic": {
            "base_url": "https://api.anthropic.com/v1",
            "api_key_env": "ANTHROPIC_API_KEY"
        },
        "mistral": {
            "base_url": "https://api.mistral.ai/v1",
            "api_key_env": "MISTRAL_API_KEY"
        },
        "gemini": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "api_key_env": "GEMINI_API_KEY"
        },
        "cohere": {
            "base_url": "https://api.cohere.com/v1",
            "api_key_env": "COHERE_API_KEY"
        },
        "xai": {
            "base_url": "https://api.x.ai/v1",
            "api_key_env": "XAI_API_KEY"
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY"
        },
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "api_key_env": None  # 使用任何字符串都可以
        },
        "bailian": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key_env": "DASHSCOPE_API_KEY"
        },
        "vllm": {
            "base_url": "http://localhost:8000/v1",
            "api_key_env": None
        }
    }

    def __init__(self, provider="openai", api_key=None, temperature: float = 0.7):
        """
        初始化 LLM 客户端
        :param base_url: API 地址，例如 "http://localhost:8000/v1"
        :param api_key: API key（Ollama 等兼容实现可用任意字符串）
        :param model: 模型名称，例如 "qwen3" 或 "gpt-4o-mini"
        :param temperature: 采样温度（[0-1]，数值越大随机性越高）
        """
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")

        self.provider = provider
        self.temperature = temperature  # 保存默认 temperature
        config = self.PROVIDERS[provider]

        if not api_key and config["api_key_env"]:
            import os
            api_key = os.getenv(config["api_key_env"])
            if not api_key:
                raise ValueError(f"Missing API key, please set {config['api_key_env']}")

        if not api_key:
            api_key = "none"

        self.client = OpenAI(api_key=api_key, base_url=config["base_url"])

    def chat(self, model, messages, stream=False, **kwargs):
        """
        普通调用: 返回字符串
        流式调用: 返回生成器，每次 yield 新的 token
        """
        # 如果用户没有传 temperature，则用默认的
        # top_p、max_tokens 这些常用参数也可以通过kwargs设置
        if "temperature" not in kwargs:
            kwargs["temperature"] = self.temperature
        if stream:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **kwargs
            )

            def generator():
                for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta

            return generator()
        else:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return resp.choices[0].message.content


if __name__ == '__main__':
    model_name = "qwen3.5"
    llm = LLMClient(provider="ollama")

    print("流式模式结果：", end="", flush=True)
    for token in llm.chat(
            model=model_name,
            messages=[{"role": "user", "content": "用少于1000个字总结《西游记》"}],
            stream=True
    ):
        print(token, end="", flush=True)

    print()  # 换行

    result = llm.chat(model_name, [{"role": "user", "content": "用李白的风格给我写一首描写饮酒的诗"}])
    print("普通模式结果：\n", result)
