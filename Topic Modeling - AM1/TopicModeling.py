# Data manipulation libraries
import pandas as pd
import numpy as np
import re
import string
import os

# Data visualization libraries
import matplotlib.pyplot as plt
#%matplotlib inline
import matplotlib
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
import spacy
import gensim
from gensim import corpora

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import os

News = []
Type = []

# Ruta local a tu carpeta de datos
path = "data"

# Subcarpetas con las categorías
folders = ["business", "entertainment", "politics", "sport", "tech"]

for category in folders:
    folder_path = os.path.join(path, category)
    files = os.listdir(folder_path)

    for text_file in files:
        file_path = os.path.join(folder_path, text_file)

        # Abre como texto normal (UTF-8)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            data = f.read()

        News.append(data)
        Type.append(category)

data = {'news': News, 'type': Type}
news_df = pd.DataFrame(data)

# check the first 5 rows from dataset

news_df.head()

# check the randomly 5 rows from dataset

news_df.sample(5)

# finding out how many rows and columns in our dataset

news_df.shape

# check information about all columns

news_df.info()

# change the Dtype of type column

news_df['type'] = news_df['type'].astype('category')
news_df.info()

# describe the dataset

news_df.describe()

# check the duplicate values in dataset

news_df.duplicated().sum()

# remove the duplicate value and check the new shape of dataset

news_df = news_df.drop_duplicates()
news_df.shape

# check the null or missing values

news_df.isna().sum()

# create new data frame from original dataset for further data analysis.

df = news_df.copy()

# check the distribution of type column

df['type'].value_counts().reset_index()

# check the distribution of different types of Articles in the dataset

fig = px.histogram(df, x='type', color='type')
fig.update_layout(xaxis_title='News Type', yaxis_title='Total Articles')
fig.show()

# add new column length of the each article of news column

df['length']=df['news'].apply(len)

# add new column of word count of each article

df['word_count'] = df['news'].apply(lambda x: len(str(x).split(" ")))
df.head()

# Total number of words present in the whole corpus

Total_words=sum(df['word_count'])
Total_words

# Distribution of Articles Length of different news type

plt.figure(figsize=(14,6))
sns.kdeplot(data=df, x=df['length'], hue=df['type'])
plt.title('Distribution of Articles Length in each news category', color='black', fontsize=14)
plt.xlabel('Length of Articles', color='black', fontsize=14)
plt.ylabel('Density', color='black', fontsize=14)
plt.tight_layout()
plt.show()

types_article = df['type'].unique()

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(20,16))

for i, article in enumerate(types_article):
    ax = axs[i//2, i%2]
    sns.histplot(x=df[df['type']==article]['word_count'], kde=True, ax=ax)
    ax.set_title(f'Distribution of {article} article', size=20, color='red')
    ax.set_xlabel('Length of total words', fontsize=14, color='black')

plt.tight_layout()
plt.show()

# decode text data
#df['news'] = df['news'].apply(lambda x: x.decode('utf-8', 'ignore'))

# define a function for top N words of all articles

import nltk

nltk.download('stopwords')


def get_top_n_words(n_top_words, count_vectorizer, text_data):
    '''
    returns a tuple of the top n words in a sample and their
    accompanying counts, given a CountVectorizer object and text sample
    '''
    vectorized_headlines = count_vectorizer.fit_transform(text_data.values)
    vectorized_total = np.sum(vectorized_headlines, axis=0)
    word_indices = np.flip(np.argsort(vectorized_total)[0, :], 1)
    word_values = np.flip(np.sort(vectorized_total)[0, :], 1)

    word_vectors = np.zeros((n_top_words, vectorized_headlines.shape[1]))
    for i in range(n_top_words):
        word_vectors[i, word_indices[0, i]] = 1

    words = [word[0] for word in count_vectorizer.inverse_transform(word_vectors)]

    return (words, word_values[0, :n_top_words].tolist()[0])

# plot a bar graph of top 15 words after removing basic nltk English stopwords.

from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(stop_words='english')
words, word_values = get_top_n_words(n_top_words=15,
                                     count_vectorizer=count_vectorizer,
                                     text_data=df['news'])

fig, ax = plt.subplots(figsize=(16,8))
ax.bar(range(len(words)), word_values)
ax.set_xticks(range(len(words)))
ax.set_xticklabels(words, rotation='vertical')
ax.set_title('Top words in headlines dataset (excluding stop words)')
ax.set_xlabel('Word')
ax.set_ylabel('Number of occurences')
plt.show()

from wordcloud import WordCloud

# define function of generate word clouds for each topic to visualize

def generate_wordclouds(df, types):
    for topic_type in types:
        allWords = ' '.join([topic for topic in df[df['type']==topic_type]['news']])
        wordCloud = WordCloud(width=500, height=300, background_color="white", random_state=21, max_font_size=110).generate(allWords)
        plt.figure(figsize=(15,10))
        plt.imshow(wordCloud, interpolation="bilinear")
        plt.axis('off')
        plt.title(topic_type + ' Word Cloud')
        plt.show()

# apply the function

types = ['business', 'tech', 'sports', 'politics', 'entertainment']
generate_wordclouds(df, types)

#%%
# decode utf-8

news_df['news'] = news_df['news'].apply(lambda x: x.decode('utf-8', 'ignore'))

# here's a new function clean_text that applies the 10 text preprocessing steps to clean the texts of news column

import re
import string


def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags
    pattern = re.compile('<.*?>')
    text = pattern.sub(r'', text)

    # Remove URLs
    pattern = re.compile(r'https?://\S+|www\.\S+')
    text = pattern.sub(r'', text)

    # Replace newline characters with spaces
    text = text.replace('\n', ' ')

    # Replace non-alphabetic characters with spaces
    text = re.sub("[^a-zA-Z]", " ", text)

    # remove text within brackets
    text = re.sub(r'\([^()]*\)', '', text)

    # remove 'b' at the beginning of article
    text = re.sub(r'^b', '', text)

    # Remove punctuation
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    # remove double characters
    text = re.sub(r'\s([a-zA-Z]{2})\s', ' ', text)

    return text

# here we dont need to correct the words spelling. so we dont use TextBlob function.

# def correct_spelling(text):
#     blob = TextBlob(text)
#     return str(blob.correct())

news_df['news'] = news_df['news'].apply(clean_text)

news_df['news'][0]

# import necessary libraries for stopwords

nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords

#!pip install -U spacy
#!python3 -m spacy download en_core_web_sm

import spacy

# Get NLTK's English stop words
s = set(stopwords.words('english'))

# Add additional stop words
additional_stop_words = ['said', 'told', 'called', 'use', 'know', 'came', 'based', 'way', 'added', 'including', 'got']
s.update(additional_stop_words)

# Use the updated set of stop words in your code
len(s)


# define function to remove stopwords

def remove_stopwords(text):
    new_text = []

    for word in text.split():
        if word in s:
            new_text.append('')
        else:
            new_text.append(word)
    x = new_text[:]
    new_text.clear()
    return " ".join(x)

# load the spaCy English language model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# get the list of spaCy English stopwords
stop_words = nlp.Defaults.stop_words
len(stop_words)

len(stop_words)


# %%
# define function of remove_spacy_stopwords

def remove_spacy_stopwords(text):
    new_text = []

    for word in text.split():
        if word in stop_words:
            new_text.append('')
        else:
            new_text.append(word)
    x = new_text[:]
    new_text.clear()
    return " ".join(x)

# apply both stopwords function to remove stopwords

news_df['news'] = news_df['news'].apply(remove_stopwords)
news_df['news'] = news_df['news'].apply(remove_spacy_stopwords)

# check the news column

news_df['news'][0]

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# creating fuctions for Lemmatization and tokenization

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ']):
    output = []
    for sent in texts:
        doc = nlp(sent)
        output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return output

# make new list of texts and apply lemmatization function.

text_list = news_df['news'].tolist()

tokenized_text = lemmatization(text_list)

# check the tokenized_text

tokenized_text[0]

from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

# create CountVectorizer instance with ngram_range=(1,3)1
vectorizer = CountVectorizer(ngram_range=(1,3))

# fit the vectorizer to the corpus
vectorizer.fit(news_df['news'])

# transform the corpus into BoW matrix
bow_matrix = vectorizer.transform(news_df['news'])

from sklearn.feature_extraction.text import TfidfVectorizer

# Tf-Idf vectoriser
vectorizer = TfidfVectorizer(min_df = 0.03)
document_term_matrix = vectorizer.fit_transform(news_df['news'])

bow_matrix.shape

document_term_matrix.shape

from sklearn.decomposition import LatentDirichletAllocation

# LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=42,max_iter=100,n_jobs=-1)
lda.fit(document_term_matrix)

# LDA model
top_lda=lda.fit_transform(document_term_matrix)

print(top_lda.shape)

from wordcloud import WordCloud

vocab = vectorizer.get_feature_names_out()

# Generate a word cloud image for given topic
def word_cloud_lda(index):
  imp_words_topic=""
  comp=lda.components_[index]
  vocab_comp = zip(vocab, comp)
  sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:50]
  for word in sorted_words:
    imp_words_topic=imp_words_topic+" "+word[0]

  wordcloud = WordCloud(width=600, height=400,max_font_size=100).generate(imp_words_topic)
  plt.figure( figsize=(5,5))
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.tight_layout()
  plt.show()

# print word clouds for each topic using LSA

for i in range(5):
    word_cloud_lda(i)
