#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 刘立军
# @date    : 2025-09-11
# @description: 解析查询graphml文件

from cachetools import LRUCache
import networkx as nx
import math

file_cache = LRUCache(maxsize=100)

import os
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
data_dir = os.path.join(current_dir,"assert")

def build_echarts_data(nodes=[], edges=[],categories=set()):
    result = {
        "nodes": nodes,
        "links":edges,
        "categories": categories
    }

    return result

def get_graph(file_id:str):
    """获取图数据"""
    if file_id in file_cache:
        content = file_cache[file_id]
    else:
        fp = os.path.join(data_dir,f"{file_id}.graphml")
        if not os.path.isfile(fp):
            fp = os.path.join(data_dir,"demo.graphml")
        content = nx.read_graphml(fp)
        file_cache[file_id] = content
    return content

def get_root(file_id:str,page=1, limit=20):
    """获取根节点和对应的边"""
    G = get_graph(file_id)

    # 确保是有向图
    if not G.is_directed():
        G = G.to_directed()

    # 找到所有无入边的顶层节点
    top_nodes = [n for n in G.nodes if G.in_degree(n) == 0]

    if not top_nodes:
        return {"data":build_echarts_data(),"total_page":0}
    
    # 分页
    total = len(top_nodes)
    total_page = math.ceil(total/limit)
    start_idx = (page - 1) * limit
    if start_idx >= total:
        return {"data":build_echarts_data(),"total_page":total_page}
    end_idx = start_idx + limit    
    if total >= end_idx:
        top_nodes_page = top_nodes[start_idx:end_idx]
    else:
        top_nodes_page = top_nodes[start_idx:]

    # print("顶层节点（无入边）:", top_nodes_page)
        
    result_nodes = []
    result_links = []    
    categories_set = set()  # 类别集合

    for node_id in top_nodes_page:
        n = get_node_info(G,node_id)
        result_nodes.append(n)
        categories_set.add(n['category'])

    return {"data":build_echarts_data(result_nodes,result_links,categories_set),"total_page":total_page}

def get_node_info(G,node_id:str):
    node_data = G.nodes[node_id]
    return {
        "id": node_id,
        "name": node_data.get("id", node_id),
        "category": node_data.get("type", "Unknown")
    }

def get_child(file_id:str,node_id:str):
    """查询子节点"""
    G = get_graph(file_id)

    if node_id not in G.nodes:
        return build_echarts_data()     

    # 用于输出的结构
    result_nodes = []
    result_links = []    
    categories_set = set()  # 类别集合

    n = get_node_info(G,node_id)
    result_nodes.append(n)
    categories_set.add(n['category'])

    # 遍历一级边
    for neighbor in G.neighbors(node_id):
        edge_data = G.get_edge_data(node_id, neighbor)
        edge_label = edge_data.get("label", "") if edge_data else ""            
        result_links.append({
            "source": node_id,
            "target": neighbor,
            "label": edge_label
        })

        # 加入二级节点
        if neighbor not in result_nodes:
            n = get_node_info(G,neighbor)
            result_nodes.append(n)
            categories_set.add(n['category'])

    return build_echarts_data(result_nodes,result_links,categories_set)

if __name__ == '__main__':

    r = get_root("123")
    print(r)
    r = get_child(file_id="123",node_id="1822ba17b297d5f6a4e5e3497c08a639925cef62b4518c01c04d30b694030795")
    print(r)