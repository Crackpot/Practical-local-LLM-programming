#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @date    : 2025-09-11
# @description: 演示faiss的使用

import os
# 启用下载进度条
os.environ["TRANSFORMERS_PROGRESS_BAR"] = "1"

import numpy as np
import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ----------------------
# 1. 数据结构设计：包含metadata
# ----------------------
# 每条数据包含文本内容和相关metadata
documents = [
    {
        "text": "人工智能是计算机科学的一个分支，研究如何使机器具有智能",
        "metadata": {
            "id": "doc_001",
            "category": "人工智能基础",
            "source": "教科书",
            "publish_date": "2023-01-15"
        }
    },
    {
        "text": "机器学习是人工智能的核心技术，让计算机能从数据中学习",
        "metadata": {
            "id": "doc_002",
            "category": "机器学习",
            "source": "技术博客",
            "publish_date": "2023-03-20"
        }
    },
    {
        "text": "深度学习是机器学习的一个子领域，基于人工神经网络",
        "metadata": {
            "id": "doc_003",
            "category": "深度学习",
            "source": "论文",
            "publish_date": "2023-02-10"
        }
    },
    {
        "text": "自然语言处理专注于让计算机理解和生成人类语言",
        "metadata": {
            "id": "doc_004",
            "category": "自然语言处理",
            "source": "技术文档",
            "publish_date": "2023-04-05"
        }
    },
    {
        "text": "计算机视觉是人工智能的一个重要方向，处理图像和视频",
        "metadata": {
            "id": "doc_005",
            "category": "计算机视觉",
            "source": "教程",
            "publish_date": "2023-01-28"
        }
    }
]

# ----------------------
# 2. 初始化模型和存储路径
# ----------------------
# 加载BGE-M3模型
# 在 中文检索/相似度匹配 场景中，bge-m3 远超 all-MiniLM-L6-v2
model = SentenceTransformer(
    'BAAI/bge-m3',
    cache_folder=Path("./model")    # 模型会下载到这个目录，第一次执行会下载模型，比较慢。
)
prefix = "为这个句子生成表示以用于检索相关句子：" 
"""
在文本嵌入（Embedding）过程中添加特定前缀是指令微调（Instruction Tuning） 思想在嵌入模型中的应用，主要作用是引导模型生成更符合特定任务需求的向量表示，从而提升下游任务（如语义检索）的效果。
当一个模型需要同时支持多种任务时，前缀可以作为 “任务标识符”：
检索任务：用"为检索生成表示：..."
聚类任务：用"为聚类生成表示：..."
问答匹配：用"为匹配问题和答案生成表示：..."
前缀的本质是通过自然语言指令引导模型聚焦任务需求，这是基于大语言模型时代的典型优化手段。对于 BGE-M3 这类支持指令微调的模型，使用官方推荐的前缀能显著提升检索效果；而对于早期不支持指令的模型（如原始 BERT），前缀可能无效甚至产生干扰。
"""

# 定义存储路径
data_folder_path = Path("./data")
if not data_folder_path.exists():
    data_folder_path.mkdir(parents=True, exist_ok=True)
index_path = data_folder_path / "faiss_index_with_metadata.index"
metadata_path = data_folder_path / "metadata.json"

# ----------------------
# 3. 文本矢量化与metadata处理，存储索引及相关文件
# ----------------------
def process_documents(docs):
    # 提取文本并添加前缀
    texts = [prefix + doc["text"] for doc in docs]
    
    # 生成向量
    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    ).astype(np.float32)
    
    # 提取metadata列表（保持与向量顺序一致）
    metadatas = [doc["metadata"] for doc in docs]
    
    return embeddings, metadatas

def create_db():
    """生成矢量库并保存"""
    if index_path.exists():
        print("数据库已创建")
        return
    # 处理文档生成向量和metadata
    embeddings, metadatas = process_documents(documents)
    print(f"向量形状: {embeddings.shape}")
    print(f"metadata数量: {len(metadatas)}")

    # 创建内积索引
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"索引向量数量: {index.ntotal}")

    # 保存FAISS索引
    faiss.write_index(index, str(index_path))

    # 保存metadata（使用JSON格式）
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)
    
    print("\n数据保存完成：")
    print(f" - 索引: {index_path}")
    print(f" - Metadata: {metadata_path}")

create_db()

# ----------------------
# 4. 加载矢量库（包含metadata）
# ----------------------
def load_data():
    # 加载索引
    index = faiss.read_index(str(index_path))
    
    # 加载metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadatas = json.load(f)
    
    print("\n数据加载完成：")
    print(f" - 索引向量数量: {index.ntotal}")
    print(f" - Metadata数量: {len(metadatas)}")
    
    return index, metadatas

index, metadatas = load_data()

# ----------------------
# 5. 带metadata的语义检索
# ----------------------
def search_with_metadata(query, top_k=2):
    # 处理查询
    query_text = prefix + query
    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True
    ).astype(np.float32)
    
    # 搜索相似向量
    scores, indices = index.search(query_embedding, top_k)
    
    # 关联metadata并返回结果
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        results.append({
            "score": float(scores[0][i]),
            "metadata": metadatas[idx],
            "text": documents[idx]["text"]  # 也可以从metadata中存储text
        })
    
    return results

if __name__ == '__main__':    

    query = "什么是深度学习"
    results = search_with_metadata(query, top_k=2)

    print(f"\n查询: {query}")
    print("检索结果:")
    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"相似度得分: {result['score']:.4f}")
        print(f"文本内容: {result['text']}")
        print(f"Metadata: {json.dumps(result['metadata'], ensure_ascii=False, indent=2)}")