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

[(0, '0.002*"let" + 0.002*"min" + 0.002*"crew" + 0.002*"cancel" + 0.002*"reach" + 0.002*"newark" + 0.002*"chang" + 0.002*"sorri" + 0.002*"respons" + 0.002*"frustrat"'),
  (1, '0.002*"issu" + 0.002*"mind" + 0.002*"anniversari" + 0.002*"incred" + 0.002*"time" + 0.002*"definit" + 0.002*"dm" + 0.002*"funni" + 0.002*"wait" + 0.002*"new"'),
    (2, '0.002*"cold" + 0.002*"pretti" + 0.002*"inflight" + 0.002*"live" + 0.002*"chicago" + 0.002*"flt" + 0.002*"set" + 0.002*"fan" + 0.002*"line" + 0.002*"airplan"'), 
    (3, '0.002*"said" + 0.002*"denver" + 0.002*"enjoy" + 0.002*"fav" + 0.002*"apolog" + 0.002*"brought" + 0.002*"rt" + 0.002*"delight" + 0.002*"need" + 0.002*"listen"'), 
 (4, '0.002*"gotcha" + 0.002*"saw" + 0.002*"omg" + 0.002*"news" + 0.002*"vega" + 0.002*"mile" + 0.002*"add" + 0.002*"world" + 0.002*"plane" + 0.002*"wonder"')]