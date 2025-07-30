import hanlp
import re
import pkuseg
import csv
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

model_id = "Qwen/Qwen3-Embedding-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with open("input.txt", "r", encoding="utf-8") as f:
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
seg = pkuseg.pkuseg()

# 1. 收集所有词向量
sentence_word_list = []  # 每个元素是该句的词列表
all_word_embs = []       # 所有词的embedding

for sent in sentences:
    words = [w for w in seg.cut(sent) if w.strip()]
    sentence_word_list.append(words)
    for w in words:
        try:
            emb = tokenizer(w, return_tensors="pt", truncation=True, max_length=32).to(device)
            with torch.no_grad():
                outputs = model(**emb)
                if hasattr(outputs, "last_hidden_state"):
                    word_emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
                else:
                    word_emb = outputs[0][:, 0, :].squeeze().cpu().numpy()
            all_word_embs.append(word_emb)
        except Exception as e:
            continue

all_word_embs = np.stack(all_word_embs)
global_mean_word_emb = np.mean(all_word_embs, axis=0)

# 2. 计算每个句子的指标
epsilon = 1e-8  # 防止除零
redundancy_list = []
info_content_list = []
info_gain_list = []
sent_emb_list = []

for i, words in enumerate(sentence_word_list):
    sent = sentences[i]
    if len(words) > 0:
        word_embs = []
        for w in words:
            emb = tokenizer(w, return_tensors="pt", truncation=True, max_length=32).to(device)
            with torch.no_grad():
                outputs = model(**emb)
                if hasattr(outputs, "last_hidden_state"):
                    word_emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
                else:
                    word_emb = outputs[0][:, 0, :].squeeze().cpu().numpy()
            word_embs.append(word_emb)
        word_embs = np.stack(word_embs)
        sent_emb = np.mean(word_embs, axis=0)
    else:
        sent_emb = np.zeros(global_mean_word_emb.shape)
    info_content = np.linalg.norm(sent_emb)
    info_gain = np.linalg.norm(sent_emb - global_mean_word_emb)
    # 新冗余度定义
    redundancy = 1 - info_gain / (info_content + epsilon)
    redundancy_list.append(redundancy)
    info_content_list.append(info_content)
    info_gain_list.append(info_gain)
    sent_emb_list.append(sent_emb)

# 统计全局词频
all_words_flat = [w for words in sentence_word_list for w in words]
word_counts = Counter(all_words_flat)
total_words = sum(word_counts.values())

def word_prob(w):
    # 加1平滑，防止概率为0
    return (word_counts[w] + 1) / (total_words + len(word_counts))

def shannon_entropy(words):
    if not words:
        return 0.0
    values, counts = np.unique(words, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-12))

def extropy(words):
    if not words:
        return 0.0
    values, counts = np.unique(words, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum((1 - probs) * np.log2(1 - probs + 1e-12))

def sentence_self_information(words):
    if not words:
        return 0.0
    info = [-np.log(word_prob(w)) for w in words]
    return np.mean(info)  # 或sum(info)取总自信息

entropy_list = []
extropy_list = []
self_info_list = []
for words in sentence_word_list:
    entropy_list.append(shannon_entropy(words))
    extropy_list.append(extropy(words))
    self_info_list.append(sentence_self_information(words))

# 归一化
def minmax_norm(arr):
    arr = np.array(arr)
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())

redundancy_norm = minmax_norm(redundancy_list)
info_content_norm = minmax_norm(info_content_list)
info_gain_norm = minmax_norm(info_gain_list)
entropy_norm = minmax_norm(entropy_list)
extropy_norm = minmax_norm(extropy_list)
self_info_norm = minmax_norm(self_info_list)

with open("sentence_info_metrics.csv", "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['句子', '冗余度(归一化)', '信息量(归一化)', '信息增益(归一化)', '信息熵(归一化)', 'extropy(归一化)', '自信息(归一化)'])
    for i, sent in enumerate(sentences):
        writer.writerow([
            sent,
            redundancy_norm[i],
            info_content_norm[i],
            info_gain_norm[i],
            entropy_norm[i],
            extropy_norm[i],
            self_info_norm[i]
        ])
        print(f"句子: {sent}\n  冗余度: {redundancy_norm[i]:.3f} 信息量: {info_content_norm[i]:.3f} 信息增益: {info_gain_norm[i]:.3f} 信息熵: {entropy_norm[i]:.3f} extropy: {extropy_norm[i]:.3f} 自信息: {self_info_norm[i]:.3f}")

print("已保存到 sentence_info_metrics.csv")
