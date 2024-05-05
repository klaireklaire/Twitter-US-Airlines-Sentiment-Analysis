import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize
import pandas as pd
from ml import get_top_features

def latent_dirichlet_allocation(processed_text):
    word_dict = corpora.Dictionary([processed_text])
    text = [word_dict.doc2bow(processed_text)]

    model = LdaModel(text, num_topics=5, id2word=word_dict, passes=15)

    topics = model.print_topics(num_words=10)
    res = []
    for topic in topics:
        res.append(topic)
    return res