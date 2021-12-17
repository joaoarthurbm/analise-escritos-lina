import nltk
import csv
from nltk.corpus import stopwords
from gensim import corpora, models
import re


def read_map():
    with open('dados.csv') as csvfile:
        reader = csv.DictReader(csvfile)

        raw_data = {}

        for line in reader:
            k, v =  line['id'], line['conteudo']
            raw_data[k] = re.sub(r'[^\w\s]','',v.lower())
    
    return raw_data

def remove_stopwords(raw_data):

    clean_data = {}
    stop_words = set(stopwords.words('portuguese'))
    stops = ["á", "à", "dizer", "ainda", "é", "grande", "êle", "êla", "mesma", "parte", "quer", "sobre", "quase", "nesta", "todo", "todos", "assim", "vai", "tão"] 
    for i in stops:
        stop_words.add(i)

    

    for k,v in raw_data.items():
        filtered_text = [w for w in v.split() if not w.lower() in stop_words]
        clean_data[k] = filtered_text

    return clean_data

def extract_topics(data_map):
    texts = list(data_map.values())
    dictionary_LDA = corpora.Dictionary(texts)
    
    dictionary_LDA.filter_extremes(no_below=3)
    corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in texts]

    num_topics = 15
    lda_model = models.LdaModel(corpus, num_topics=num_topics,
                                  id2word=dictionary_LDA,
                                  passes=4, alpha=[0.01]*num_topics,
                                  eta=[0.01]*len(dictionary_LDA.keys()))

    
    topics = lda_model.print_topics(num_topics=15, num_words=10)

    for i in range(len(corpus)):
        
        
        s_out = ""
        list_topics = lda_model[corpus[i]]
        for topic in list_topics:
            id_topic = topic[0]
            words = topics[id_topic][1]
            s_out += words + "-"


        ids = list(data_map.keys())
        print('{}-{}-{}'.format(ids[i], list_topics, s_out[:-1]  ))


raw_data = read_map()
clean_data_map = remove_stopwords(raw_data)
extract_topics(clean_data_map)
