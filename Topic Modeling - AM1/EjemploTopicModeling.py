#Topic modeling, para presentacion v2

import os
import re
import string
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from wordcloud import WordCloud

import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


#Cargar los datos
#Estan subdivididos en carpetas
#Son archivos de texto, en cierta parte preprocesados de articulos de la BBC

def cargar(path="data"):
    carpetas = ["Business", "Entertainment", "Politics", "Sport", "Tech"]
    noticias = []
    tipos = []

    for category in carpetas:
        path_carpeta = os.path.join(path, category)
        archivos = os.listdir(path_carpeta)

        for f in archivos:
            path_archivos = os.path.join(path_carpeta, f)
            with open(path_archivos, "r", encoding="utf-8", errors="ignore") as file:
                text = file.read()
            noticias.append(text)
            tipos.append(category)

    df = pd.DataFrame({"noticias": noticias, "type": tipos})
    return df


df = cargar()
df = df.drop_duplicates().reset_index(drop=True)

# Preprocesamiento

def limpieza(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df["limpieza"] = df["noticias"].apply(limpieza)



# Manejo de las stopwords
# Por esto se hizo stopwords.txt

def cargar_stopwords(path="stopwords.txt"):
    if not os.path.exists(path):
        print("No se encontro stopwords.txt")
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set([w.strip() for w in f.readlines()])

stopwords = cargar_stopwords()

def quitar_stopwords(text):
    return " ".join([w for w in text.split() if w not in stopwords])

df["limpieza"] = df["limpieza"].apply(quitar_stopwords)

# Word Cloud

def hacer_wordcloud(df):
    categorias = df["type"].unique()

    for cat in categorias:
        textos = df[df["type"] == cat]["limpieza"].tolist()
        todas_palabras = " ".join(textos)

        if len(todas_palabras.strip()) == 0:
            print(f"No hay palabras suficientes para '{cat}', saltando.")
            continue

        wc = WordCloud(width=600, height=400, background_color="white").generate(todas_palabras)
        plt.figure(figsize=(8, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"WordCloud – {cat}")
        plt.show()


hacer_wordcloud(df)

# Vectorizar
# TF - IDF

vectorizer = TfidfVectorizer(min_df=0.03)
dtm = vectorizer.fit_transform(df["limpieza"])
vocab = vectorizer.get_feature_names_out()


# El modelo como tal
# LDA o Latent Dirich Allocation

lda = LatentDirichletAllocation(
    n_components=5,
    random_state=42,
    max_iter=50,
    learning_method="batch"
)
lda.fit(dtm)

# Subtopicos
subtopicos = 3
for cat in df["type"].unique():
    print(f"\nSubtópicos para {cat}:")

    docs = df[df["type"] == cat]["limpieza"]
    vectorizer_cat = TfidfVectorizer(min_df=0.03)
    dtm_cat = vectorizer_cat.fit_transform(docs)
    vocab_cat = vectorizer_cat.get_feature_names_out()

    lda_cat = LatentDirichletAllocation(n_components=subtopicos, random_state=42)
    lda_cat.fit(dtm_cat)

    for i in range(subtopicos):
        comp = lda_cat.components_[i]
        top_palabras = [vocab_cat[j] for j in comp.argsort()[-10:]]
        print(f"  Subtópico {i}: {', '.join(top_palabras)}")

    doc_probs = lda_cat.transform(dtm_cat)
    df.loc[df["type"] == cat, f"topico_sub_{cat}"] = np.argmax(doc_probs, axis=1)


# Otro wordcloud, pero este ya el bonito :D

# Sino solo salia topico 0 - 4
titulos_topicos = {
    0: "Tech",
    1: "Sports",
    2: "Entertainment",
    3: "Politics",
    4: "Buisness"
}

def lda_wordcloud(model, vocab, topic_index):
    comp = model.components_[topic_index]
    top_palabras = [vocab[i] for i in comp.argsort()[-50:]]
    text = " ".join(top_palabras)

    wc = WordCloud(width=600, height=400, background_color="white").generate(text)
    plt.figure(figsize=(8, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")

    plt.title(f"{titulos_topicos.get(topic_index, 'Desconocido')}")
    plt.show()


for topic_idx in range(5):
    lda_wordcloud(lda, vocab, topic_idx)

#Top palabras

def grafico_palabras(model, vocab, topic_index, n=10):
    comp = model.components_[topic_index]
    indices = comp.argsort()[-n:]
    palabras = [vocab[i] for i in indices]
    valores = comp[indices]

    plt.figure(figsize=(8, 4))
    plt.barh(palabras, valores)
    plt.title(f"Top {n} palabras del Tópico: {titulos_topicos.get(topic_index, 'Desconocido')}")
    plt.show()


for topic in range(5):
    grafico_palabras(lda, vocab, topic)
