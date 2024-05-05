# import gensim
# from gensim import corpora
# from gensim.models.ldamodel import LdaModel
# from nltk.tokenize import word_tokenize
# import pandas as pd
# from ml import get_top_features

# def latent_dirichlet_allocation(processed_text):
#     word_dict = corpora.Dictionary([processed_text])
#     text = [word_dict.doc2bow(processed_text)]

#     model = LdaModel(text, num_topics=5, id2word=word_dict, passes=15)

#     topics = model.print_topics(num_words=10)
#     res = []
#     for topic in topics:
#         res.append(topic)
#     return res

import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel

def latent_dirichlet_allocation():
    doc = pd.read_csv('cleaned_readable.csv')
    processed_text = doc["processed_text"]

    tokens = [word_tokenize(str(sent)) for sent in processed_text]

    word_dict = corpora.Dictionary(tokens)
    text = [word_dict.doc2bow(token) for token in tokens] 

    model = LdaModel(text, num_topics=5, id2word=word_dict, passes=15)


    topics = model.print_topics(num_words=10)

    for topic in topics:
        print(topic)

def evaluate_models_by_perplexity(start=2, limit=15, step=1):
    perplexity_values = []
    model_list = []
    doc = pd.read_csv('cleaned_readable.csv')
    processed_text = doc["processed_text"]

    processed_texts = [word_tokenize(str(sent)) for sent in processed_text]
    # print(processed_texts)
    for num_topics in range(start, limit, step):
        # Create a dictionary and corpus required for Topic Modeling
        dictionary = corpora.Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]

        # Build the LDA model
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15, random_state=100,
                         alpha='auto', eta='auto', iterations=400)

        model_list.append(model)

        # Compute Log Perplexity
        log_perplexity = model.log_perplexity(corpus)
        perplexity_values.append(log_perplexity)

    return model_list, perplexity_values


def main():
    # latent_dirichlet_allocation()
    # print(evaluate_models())
    print(evaluate_models_by_perplexity())
   
    
    
if __name__ == "__main__":
    main()