import nltk
import re

from gensim.models import Word2Vec
from nltk.corpus import stopwords


paragraph = """
Animals are an important part of nature.
Dogs are friendly animals and are loyal to humans.
Cats are playful animals and enjoy sleeping.
Lions are powerful wild animals and live in forests.
Tigers are strong animals and are skilled hunters.
Elephants are large animals with long trunks.
Monkeys live in trees and enjoy eating fruits.
Birds have wings and many birds can fly.
Fish live in water and use their fins to swim.
Cows provide milk and are raised on farms.
Dogs and cats are popular pets.
Wild animals live in forests and grasslands.
Animals need food, water, and shelter to survive.
Nature is important for protecting animals.
"""

# Text preprocessing
text = re.sub(r'\[[0-9]*\]', ' ', paragraph)
text = re.sub(r'\s+', ' ', text)
text = text.lower()
text = re.sub(r'\d', ' ', text)
text = re.sub(r'\s+', ' ', text)


# Sentence tokenization
sentences = nltk.sent_tokenize(text)


# Word tokenization
sentences = [
    nltk.word_tokenize(sentence)
    for sentence in sentences
]


# Remove stopwords

stop_words = set(stopwords.words('english'))

for i in range(len(sentences)):
    sentences[i] = [
        word for word in sentences[i]
        if word not in stop_words and word.isalpha() ]


# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)


#  Get vocabulary
words = model.wv.index_to_key


#  Find the word vector
vector = model.wv['animals']


# 9. Find the most similar words

similar = model.wv.most_similar('animals')

similar1 = model.wv.most_similar('dogs')

similar2 = model.wv.most_similar('cats')

similar3 = model.wv.most_similar('wild')


