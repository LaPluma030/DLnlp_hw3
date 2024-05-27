import gensim
from gensim.models import Word2Vec
import numpy as np
import jieba
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

file_path = r"jyxstxtqj_downcc.com"


def compute_similarity(model, word1, word2):
    if isinstance(model, Word2Vec):
        vector1 = model.wv[word1]
        vector2 = model.wv[word2]
    else:
        vector1 = model.word_vectors[model.dictionary[word1]]
        vector2 = model.word_vectors[model.dictionary[word2]]
    similarity = cosine_similarity([vector1], [vector2])[0][0]
    return similarity

def preprocess_chinese_corpus(folder_path):
    corpus = []
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', encoding='gb18030') as f:
                content = f.read()
                tokenized_sentences = [jieba.lcut(sentence) for sentence in content.split('\n') if sentence.strip()]
                corpus.extend(tokenized_sentences)
    return corpus

chinese_tokenized_sentences = preprocess_chinese_corpus(file_path)


# 训练Word2Vec模型
model = Word2Vec(chinese_tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")

# 计算余弦相似度
def print_similarity(word1, word2):
    word2vec_similarity = compute_similarity(model, word1, word2)

    print(f"Word1: {word1}, Word2: {word2}")
    print(f"Word2Vec similarity between '{word1}' and '{word2}': {word2vec_similarity}")    

print_similarity("郭靖","黄蓉")
print_similarity("杨过","小龙女")
print_similarity("张无忌","赵敏")

#词语聚类
def plot_clusters(model, n_clusters = 5, outputpath = 'cluster.png'):
    # 需要聚类的词列表
    if isinstance(model, Word2Vec):
        words = list(model.wv.index_to_key)
        word_vectors = model.wv[words]
    else:
        words = list(model.dictionary.keys())
        word_vectors = model.word_vectors

    # 初始化并训练 KMeans 聚类算法
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    kmeans.fit(word_vectors)

    # 获取聚类结果
    labels = kmeans.labels_

    # 绘制散点图
    plt.figure(figsize=(10, 10))
    for i, label in enumerate(labels):
        x, y = word_vectors[i][0], word_vectors[i][1]
        plt.scatter(x, y, c=f'C{label}')
        plt.text(x+0.03, y+0.03, words[i], fontsize=9)

    plt.title('Word Clusters Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(outputpath)
    plt.show()

plot_clusters(model)

# 计算段落语义关联
def get_paragraph_vector(paragraph, model):
    tokens = [word for word in jieba.lcut(paragraph) if word in model.wv] if isinstance(model, Word2Vec) else [word for word in jieba.lcut(paragraph) if word in model.dictionary]
    vectors = [model.wv[token] for token in tokens] if isinstance(model, Word2Vec) else [model.word_vectors[model.dictionary[token]] for token in tokens]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# 从语料库中随机选择两个句子作为示例段落
sample_paragraph1 = random.choice(chinese_tokenized_sentences)
sample_paragraph2 = random.choice(chinese_tokenized_sentences)

# 计算示例段落的语义关联
sample_paragraph1_str = ' '.join(sample_paragraph1)
sample_paragraph2_str = ' '.join(sample_paragraph2)
paragraph_vector1 = get_paragraph_vector(sample_paragraph1_str, model)
paragraph_vector2 = get_paragraph_vector(sample_paragraph2_str, model)

paragraph_similarity = cosine_similarity([paragraph_vector1], [paragraph_vector2])[0][0]
print(f"Semantic similarity between paragraphs: {paragraph_similarity}")

print(sample_paragraph1_str)
print(sample_paragraph2_str)