import re
import csv
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from neo4j import GraphDatabase
import os
import warnings

print("当前工作目录：", os.getcwd())
warnings.filterwarnings("ignore", module="torch.cuda")

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型初始化
embedding_model_id = "Qwen/Qwen3-Embedding-0.6B"
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_id)
embedding_model = AutoModel.from_pretrained(embedding_model_id, trust_remote_code=True)
embedding_model.to(device)

rerank_model_id = "Qwen/Qwen3-Reranker-0.6B"
rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_id)
rerank_model = AutoModel.from_pretrained(rerank_model_id, trust_remote_code=True)
rerank_model.to(device)

# 读取文本
with open("./EmbeddingModels/input.txt", "r", encoding="utf-8") as f:
    raw_text = f.read().strip()

def regex_sentence_split(text):
    sents = re.split(r'([。！？])', text)
    sentences = []
    for i in range(0, len(sents)-1, 2):
        sentence = sents[i].strip() + sents[i+1]
        if sentence:
            sentences.append(sentence)
    if len(sents) % 2 != 0 and sents[-1].strip():
        sentences.append(sents[-1].strip())
    return sentences

sentences = regex_sentence_split(raw_text)

# Neo4j配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "88888888"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        if hasattr(outputs, "last_hidden_state"):
            emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        else:
            emb = outputs[0][:, 0, :].squeeze().cpu().numpy()
    return emb

def rerank_score(candidate, context):
    inputs = rerank_tokenizer(candidate, context, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = rerank_model(**inputs)
        # 这里假设模型输出logits用于评分，具体看模型实际返回结构
        if hasattr(outputs, "logits"):
            score = outputs.logits.squeeze().cpu().item()
        else:
            # fallback: 用CLS向量norm做简易评分
            if hasattr(outputs, "last_hidden_state"):
                cls_vec = outputs.last_hidden_state[:, 0, :].squeeze()
                score = torch.norm(cls_vec).cpu().item()
            else:
                score = 0.0
    return score

def insert_entity(tx, name):
    tx.run("MERGE (e:Entity {name: $name})", name=name)

def generate_candidates(sentence, min_len=1, max_len=4):
    candidates = []
    length = len(sentence)
    for start in range(length):
        for end in range(start + min_len, min(start + max_len, length) + 1):
            candidate = sentence[start:end]
            candidates.append(candidate)
    return candidates

threshold_score = -float('inf')  # 你可以设置一个阈值过滤低分，暂时不开启

with driver.session() as session:
    for idx, sent in enumerate(sentences):
        print(f"\n处理第{idx+1}句：{sent}")
        candidates = generate_candidates(sent)
        candidates = [c for c in candidates if c.strip()]
        scored = []
        for c in candidates:
            score = rerank_score(c, sent)
            if score > threshold_score:
                scored.append((c, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_entities = [c for c, s in scored[:5]]  # 取top5实体候选

        print(f"  Top实体候选: {top_entities}")

        for ent in set(top_entities):
            session.execute_write(insert_entity, ent)

print("实体抽取完成，写入Neo4j。")
