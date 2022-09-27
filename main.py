from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
import csv
from nltk.corpus import stopwords
from gensim import corpora, models
import re
import pandas as pd


import unicodedata

def remover_combinantes(string):

    string = unicodedata.normalize('NFD', string)
    return u''.join(ch for ch in string if unicodedata.category(ch) != 'Mn')

def read_map():
    with open('dados.csv') as csvfile:
        reader = csv.DictReader(csvfile)

        raw_data = {}

        for line in reader:
            k, v =  line['id'], line['conteudo']
            v = 'baiano' if v.lower() == "bahiano" else v.lower()
            raw_data[k] = remover_combinantes(re.sub(r'[^\w\s]','',v))
    
    return raw_data

def remove_stopwords(raw_data):

    clean_data = {}
    stop_words = set(stopwords.words('portuguese'))
    
    stop_domain = []
    with open('stop.txt') as file:
        stop_domain = file.readlines()
        stop_domain = [line.rstrip() for line in stop_domain]

    for i in stop_domain:
        stop_words.add(i)

    for k,v in raw_data.items():
        filtered_text = [w for w in v.split() if not w.lower() in stop_words]
        clean_data[k] = filtered_text

    return clean_data

def extract_topics(data_map, n_topics = 15, n_words = 10):
    texts = list(data_map.values())
    
    dictionary_LDA = corpora.Dictionary(texts)
    
    dictionary_LDA.filter_extremes(no_below=3)
    corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in texts]

    lda_model = models.LdaModel(corpus, num_topics=n_topics,
                                  id2word=dictionary_LDA,
                                  passes=4, alpha=[0.01]*n_topics,
                                  eta=[0.01]*len(dictionary_LDA.keys()))

    
    topics = lda_model.print_topics(num_topics=n_topics, num_words=n_words)

    for i in range(len(corpus)):
        
        s_out = ""
        list_topics = lda_model[corpus[i]]
        for topic in list_topics:
            id_topic = topic[0]
            words = topics[id_topic][1]
            s_out += words + "-"


        ids = list(data_map.keys())
        print('{}-{}-{}'.format(ids[i], list_topics, s_out[:-1]  ))

    # print topics
    for topic in topics:
        print(topic)

def analyze_n_grams(id_text, tokens, from_n, to_n, top):
    data = [' '.join(tokens)]
    
    # Getting trigrams
    vectorizer = CountVectorizer(ngram_range = (from_n,to_n))
    vectorizer.fit_transform(data)
    features = (vectorizer.get_feature_names_out())

    # Applying TFIDF
    vectorizer = TfidfVectorizer(ngram_range = (from_n,to_n))
    X2 = vectorizer.fit_transform(data)

    # Getting top ranking features
    sums = X2.sum(axis = 0)
    term_rank = []
    for col, term in enumerate(features):
        term_rank.append( (id_text, term, sums[0,col] ))
    ranking = pd.DataFrame(term_rank, columns = ['id_texto', 'term','rank'])
    words = (ranking.sort_values('rank', ascending = False))
    print (words.head(20))

# returns an unified list of tokens
def merge_texts_to_blob(text_ids, clean_data_map):
    blob = []
    for k in text_ids:
        tokens = clean_data_map[k]
        blob.extend(tokens)

    return blob

# preprocessing
raw_data = read_map()

# map<string,[tokens]>
clean_data_map = remove_stopwords(raw_data)

#cronicas = ["tp_dn1_cultura_e_nao_cultura",
#        "tp_dn2_arquitetura_ou_arquitetura", "tp_dn3_inatualidade_da_cultura",
#        "tp_dn4_a_escola_e_a_vida",
#        "tp_dn5_casas_ou_museus",
#        "tp_dn6_a_invasao",
#        "tp_dn7_a_lua",
#        "tp_dn8_arte_industrial"]

#tokens_cronicas = []
#for k in cronicas:
#    if k in clean_data_map:
#        tokens_cronicas.extend(clean_data_map[k])
#        print(tokens_cronicas)
#extract_topics({'cronicas': tokens_cronicas}, 5, 10)

### ngrams analysis - one by one
# just the 20 most important and 1, 2 e 3grams
#for k, v in clean_data_map.items():
#    analyze_n_grams(k, v, 1, 1, 20)
#    print()

### ngrams analysis - the entire corpus
text_blob = []
texts = list(clean_data_map.values())
for i in range(len(texts)):
    for token in texts[i]:
        text_blob.append(token)

stemmer = nltk.stem.RSLPStemmer()
text_blob = [stemmer.stem(palavra) for palavra in text_blob]

for i in [1]:
    analyze_n_grams("blob", text_blob, i, i, 20)



#analyze_n_grams("ob", merge_texts_to_blob([c for c in clean_data_map.keys() if c.startswith("ob")], clean_data_map), 1, 1, 20)
