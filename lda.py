import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize
import pandas as pd

doc = pd.read_csv('cleaned_readable.csv')
processed_text = doc["processed_text"]

tokens = [word_tokenize(str(sent)) for sent in processed_text]

word_dict = corpora.Dictionary(tokens)
text = [word_dict.doc2bow(token) for token in tokens] 

model = LdaModel(text, num_topics=5, id2word=word_dict, passes=15)


topics = model.print_topics(num_words=10)

for topic in topics:
    print(topic)
