import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
from bs4 import BeautifulSoup
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import altair as alt
import itertools
from pyvis.network import Network


def get_soup(url):
    response = requests.get(url)
    #response.encoding = response.apparent_encoding
    html = response.text
    return BeautifulSoup(html, 'html.parser')

def minmax_norm(df):
    df_output = (df - df.min()) / ( df.max() - df.min())
    return df_output.fillna(0)

def remove_all_zero_col(df):
    df = df.copy()
    for col in df.columns:
        if (df[col] == 0).all():
            df.drop(col, axis=1, inplace=True)
    return df

@st.cache
def create_df():
    url = "https://translation-word-order-api.herokuapp.com/langs"
    soup = get_soup(url)

    distance_dict = {}
    for langs, distance in json.loads(soup.text).items():
        src_lang, tgt_lang = langs.split("_")
        if src_lang not in distance_dict:
            distance_dict[src_lang] = {}
        distance_dict[src_lang][tgt_lang] = float(distance)
    df = pd.DataFrame(distance_dict)
    return df.fillna(0)


@st.cache
def create_pca_df(df):
    x = remove_all_zero_col(minmax_norm(df)).T

    pca = PCA(n_components=2)
    pca.fit(x)
    x_2d = pca.transform(x)
    df_2d = pd.DataFrame(x_2d)
    df_2d.index = x.index
    df_2d.columns = ["PC1", "PC2"]
    return df_2d

def create_lang_network(x, cluster_by_lang):
    network = Network(notebook=True, width="100%")
    size = 10
    thresh = 0.9
    physics = True

    for lang1, lang2 in itertools.combinations(list(x.T.columns), 2):
        network.add_node(lang1, size=size, physics=physics, group=str(cluster_by_lang[lang1]))
        network.add_node(lang2, size=size, physics=physics, group=str(cluster_by_lang[lang2]))
        vec_lang1, vec_lang2 = x.loc[lang1], x.loc[lang2]
        dist = np.dot(vec_lang1, vec_lang2) / (np.linalg.norm(vec_lang1) * np.linalg.norm(vec_lang2))
        if dist < thresh: continue
        width = (1.0 - dist) * 10
        network.add_edge(lang1, lang2, width=width, title=dist)
    return network


st.set_page_config(page_title="Word Order Analysis", layout="wide")
st.title("語順による言語間距離")

description = """
#### 方法
1. [Wikipedia:ウィキペディアについて](https://ja.wikipedia.org/wiki/Wikipedia:%E3%82%A6%E3%82%A3%E3%82%AD%E3%83%9A%E3%83%87%E3%82%A3%E3%82%A2%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6) の各言語バージョンの本文を収集
1. 収集した各言語の文を https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt により翻訳（52言語x52言語）
1. 翻訳モデルの単語にかかるアテンションの重みにより翻訳時の単語の並び替えを検出 (参考 https://qiita.com/sentencebird/items/6f4ef30187d329a95543)
1. 翻訳元言語に対して、各言語における並び替えの量を計算しベクトル化する
1. 翻訳元言語のベクトル同士を比較（コサイン類似度）して、言語間の距離を計算する
"""
st.markdown(description)

df = create_df()
x = remove_all_zero_col(minmax_norm(df)).T

st.header("並び替えの大きさの行列")
description = """
行が翻訳元、列が翻訳先の言語に対応した語順の並び替えの大きさを示した行列

各行で正規化済み（0~1）
"""
st.markdown(description, unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(20,20))
x = remove_all_zero_col(minmax_norm(df)).T
sns.heatmap(x, ax=ax, square=True, cbar=False)
ax.set_xlabel("Target Language")
ax.set_ylabel("Source Language")
fig.show()
with st.spinner("Creating a matrix ..."):
    st.pyplot(fig)

    
df_2d = create_pca_df(df)

# クラスタリング
cls = KMeans(n_clusters=6)
clusters = cls.fit(x)
clusters_df = pd.Series(clusters.labels_, index=df_2d.index)


st.header("主成分空間")
description = """
1プロットが翻訳元言語に対応

上記Matrixの行ベクトル（52次元）を主成分空間に写像
"""
st.markdown(description, unsafe_allow_html=True)
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(df_2d.PC2, df_2d.PC1)
for i, country in enumerate(df_2d.index):
    ax.annotate(country, (df_2d.iloc[i].PC2, df_2d.iloc[i].PC1))

with st.spinner("Creating a plot ..."):
    st.plotly_chart(fig)
    #st.pyplot(fig, use_container_width=True)

st.header("各言語の距離を示したネットワーク")    
description = """
各翻訳元言語間のベクトルの類似度に応じて、類似度の大きな言語を結んだネットワーク図

色は「並び替えの大きさの行列」のベクトルをクラスタリングして割り振ったクラスタに対応
"""
st.markdown(description, unsafe_allow_html=True)
with st.spinner("Creating a network ..."):
    n = create_lang_network(x, clusters_df)
    n.show(f"output.html")

    html_file = open(f"output.html", 'r', encoding='utf-8')
    source_code = html_file.read() 
    components.html(source_code, height=1200, width=1000)