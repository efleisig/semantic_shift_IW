# -*- coding: utf-8 -*-
"""
Code for the project "Automatic Detection of Semantic Shift in Spanish with 
Context Optimization".

This project downloads Google n-gram lists, creates vector embeddings for the 
words in a given n-gram dataset, and trains and tests a classifier to detect 
whether a word underwent semantic shift between two time periods.

@author: Eve Fleisig
"""

import numpy as np
from collections import OrderedDict
from nltk.stem  import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from datetime import datetime
from google_ngram_downloader import readline_google_store
import pickle
import gzip
import os
import random

# For downloads
from py.path import local
from google_ngram_downloader.util import iter_google_store
from string import ascii_lowercase
from itertools import product, chain
import requests

# For classifier
from sklearn.svm import SVC
from diachrony_master.algos import ProcrustesAligner # Ported from github.com/wadimiusz/diachrony_for_russian
from diachrony_master.utils import load_model
from sklearn import metrics


# Given Google n-gram lists in `directory`, creates and returns a dictionary of
# preprocessed n-grams from that time period.
def create_ngram_lists_from_server(unigram_dt, ngram_dt, ngram_size, 
                                   max_ngrams, directory = "downloads/google_ngrams/5"):
    
    stop_words = set(stopwords.words('spanish'))
    lemmatizer=WordNetLemmatizer()    
    
    file_counter = 0
    for fname in os.listdir(directory):
        
        
        if fname.endswith(".gz"):
            print(fname, flush=True)
            
            file_counter+=1
            line_counter = 1
            
            with gzip.open(directory + "/" + fname,'rt') as f:
                
                for record in f:
                    
                    line_counter += 1
                    if line_counter > max_ngrams:
                        break
                        
                    line = record.split()
                    
                    # Lemmatize words and remove Google-specific endings
                    words = []
                    for word in line[:-3]:
                        
                        if "_" in word[1:]:
                            word= word[:word.index("_")]
                        
                        if len(word) > 0 and not word[0].istitle() and word not in stop_words:
                            word = word.lower()
                            word = lemmatizer.lemmatize(word)
                            
                            if len(word) > 0:
                                words.append(word)
                    
                    year = int(line[-3])
                    freq = int(line[-2])
                    
                    
                    # Create ngram list for each time period
                    if len(words)>1:
                        for date_range in ngram_dt:
                            if year > date_range[0] and year < date_range[1]:
                                
                                for i in range(int(freq)):
                                    ngram_dt[date_range].append(words)
                                    
                                break
                            
        if file_counter % 5 == 0:                    
            with open('ngram_dict_'+ str(ngram_size) + '_through_' + fname + '.pickle', 'wb') as handle:
                pickle.dump(ngram_dt, handle)
            
    with open('ngram_dict_'+ str(ngram_size) +'_final.pickle', 'wb') as handle:
            pickle.dump(ngram_dt, handle)
        
    return ngram_dt
    
    
# Creates Word2vec models for each time period in `ngram-dt`.
def make_word_embeddings(ngram_dt, ngram_size):
          
    # Create custom word embeddings (using word2vec)
    # Then pass them into the testing software
    models = []
    for date_range in ngram_dt:
        print(date_range, " Number of ngrams: ", len(ngram_dt[date_range]), flush=True)
        
        model = Word2Vec(min_count=3, size = 200, window = ngram_size)
        model.build_vocab(ngram_dt[date_range])
        model.train(ngram_dt[date_range], total_examples=model.corpus_count, epochs=model.iter, report_delay = 180)
        
        models.append(model)
        model.save("word2vec_" + str(ngram_size) + "gram_" + str(date_range[0]) + ".model")
        wv = model.wv
        wv.save("word2vec_" + str(ngram_size) + "gram_" + str(date_range[0]) +  "kv.model")


# Graphs the words most similar to`word` using t-SNE.
def graph_word_embedding(ngram_dt, word, clean_word, ngram_size, size=200):

    for date_range in ngram_dt:
        
        try:
            model = Word2Vec.load(("word2vec_" + str(ngram_size) + "gram_" + str(date_range[0]) + ".model"))
        except IOError:
            continue
        
        words=model.wv
        
        if word not in words.vocab:
            print("Not in vocab")
            continue
        
        arr = np.empty((0,size), dtype='f')
        word_labels = [word]
        close_words = model.wv.similar_by_word(word)
        arr = np.append(arr, np.array([model[word]]), axis=0)
        
        for wrd_score in close_words:
            wrd_vector = model[wrd_score[0]]
            word_labels.append(wrd_score[0])
            arr = np.append(arr, np.array([wrd_vector]), axis=0)
            
        tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(arr)
        
        with open("word2vec_"+ str(ngram_size) + "_" + str(date_range[0]) + "_" + clean_word + "_coords"  +".pickle", 'wb') as handle:
            pickle.dump(Y, handle)
        
        with open("word2vec_"+ str(ngram_size) + "_" + str(date_range[0]) + "_" + clean_word + "_labels"  +".pickle", 'wb') as handle:
            pickle.dump(word_labels, handle)
            
        x_coords = Y[:, 0]
        y_coords = Y[:, 1]
        plt.figure()
        plt.scatter(x_coords, y_coords)
        
        for label, x, y in zip(word_labels, x_coords, y_coords):
                plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        
        plt.savefig("word2vec_"+ str(ngram_size) + "_" + str(date_range[0]) + "_" + clean_word + "_"  +".png")

   
# Saves keyed vectors from Gensim model
def save_keyed_vectors(fname):
    print("in lm")
    for date_range in ngram_dt:
        print(1)
        try:
            model = Word2Vec.load(fname)
        except IOError:
            print(str(date_range[0]) + "not found")
            continue
    
        wv = model.wv
        wv.save("word2vec_" + str(max_ngrams) + "_" + str(date_range[0]) + "kv.model")
        

# Download a specific Google n-gram file.    
def test_specific_download(ngram_len=4,
                           output='downloads/google_ngrams/{ngram_len}',
                           verbose=False, lang='spa'):
    
    
    letter_indices = ((''.join(i) for i in product(ascii_lowercase, ascii_lowercase + '_')))
    if ngram_len == 5:
        letter_indices = (l for l in letter_indices if l != 'qk')
    
    output = local("/scratch/network/efleisig/downloads/google_ngrams/test_" + str(ngram_len))
    output.ensure_dir()
    
    for fname, url, request in iter_google_store_custom(ngram_len, lang=lang):
        if fname in os.listdir('downloads/google_ngrams/{ngram_len}'.format(ngram_len=ngram_len)):
            print("File already here", flush=True)
        else:
            with output.join(fname).open('wb') as f:
                for num, chunk in enumerate(request.iter_content(1024)):
                    f.write(chunk)
                

def iter_google_store_custom(ngram_len, lang="eng", indices=None):
    """Iterate over the collection files stored at Google.
    :param int ngram_len: the length of ngrams to be streamed.
    :param str lang: the langueage of the ngrams.
    :param iter indices: the file indices to be downloaded.
    :param bool verbose: if `True`, then the debug information is shown to `sys.stderr`.
    
    Ported from https://github.com/dimazest/google-ngram-downloader.
    """

    session = requests.Session()

    indices = get_indices(ngram_len) if indices is None else indices

    for index in indices:
        fname = "googlebooks-spa-all-4gram-20120701-" + index + ".gz"

        url = "http://storage.googleapis.com/books/ngrams/books/googlebooks-spa-all-4gram-20120701-" + index + ".gz"


        request = session.get(url, stream=True)
        if request.status_code != 200:
            print("Couldn't get file" + fname)
            continue

        yield fname, url, request


def get_indices(ngram_len):
    """Generate the file indeces depening on the ngram length, based on version 20120701.
    For 1grams it is::
        0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l m n o other p pos
        punctuation q r s t u v w x y z
    For others::
        0 1 2 3 4 5 6 7 8 9 _ADJ_ _ADP_ _ADV_ _CONJ_ _DET_ _NOUN_ _NUM_ _PRON_
        _PRT_ _VERB_ a_ aa ab ac ad ae af ag ah ai aj...
    Note, there is not index "qk" for 5grams.
    See http://storage.googleapis.com/books/ngrams/books/datasetsv2.html for
    more details.
    """

    if ngram_len == 1:
        letter_indices = ascii_lowercase

    else:
        letter_indices = ((''.join(i) for i in product(ascii_lowercase, ascii_lowercase + '_')))
        if ngram_len == 5:
            letter_indices = (l for l in letter_indices if l != 'qk')

    return chain(letter_indices)   
    
# Creates a single matrix of the form [word p1 p2 procrustes_score gt]
# to be split into train and test sets
def get_word_vectors(fpath, periods_list, ngram_size):
 
    models = {}
    for pd in periods_list:
        model_name = "word2vec_" + str(ngram_size) + "gram_"+ str(pd[0]) + "kv.model"
        models[pd] = load_model(model_name)
    
    word_mtx_changed = []
    word_mtx_unchanged = []
    
    file = pickle.load( open(fpath, "rb" ) )
    for entries in file:
        word = entries[0]
        
        prev_pd = 0                 # 0 = nothing prev
        num_in_pd = 0
        cur_word_rows = []
        for year in entries[1:]:
            year = int(year)
            
            cur_pd = get_period(year, periods_list)
            
            # If not first change in time period, add change from current to all later time periods
            if cur_pd == prev_pd:
                
                if num_in_pd >= 2:
                    continue
                
                num_in_pd += 1
                pds_after = [pd for pd in periods_list if pd[0] > cur_pd[0]]
                
                changed_rows = get_changed_rows(word, [cur_pd], pds_after, models)
                if changed_rows != None:
                    cur_word_rows += [row for row in changed_rows if row not in cur_word_rows]
            
            # If first change in time period, add change from all previous to current and all later time periods
            else:
                num_in_pd = 1
                
                pds_before = [pd for pd in periods_list if pd[0] < cur_pd[0]]
                pds_cur_after = [pd for pd in periods_list if pd[0] >= cur_pd[0]]
                
                changed_rows = get_changed_rows(word, pds_before, pds_cur_after, models)
                if changed_rows != None:
                    cur_word_rows += [row for row in changed_rows if row not in cur_word_rows]
                
            prev_pd = cur_pd
        
        # Add rows for all valid period combos where there was no change
        word_mtx_changed += cur_word_rows
        word_mtx_unchanged += get_unchanged_rows(word, cur_word_rows, periods_list, models) 
        
    for index, row in enumerate(word_mtx_changed):
        if row[:-1] == word_mtx_unchanged[index][:-1]:
            print("Sanity check failed: row", row, "in both matrices")
        
    return word_mtx_changed, word_mtx_unchanged
        
    
# Returns the time period that a year belongs to.
def get_period(year, period_dt):
    
    if year < 1522:
        return (1522, 1899)
    for date_range in period_dt:
        if year > date_range[0] and year < date_range[1]:
            
            return date_range
    
    return (2005, 2009)


# Adds rows for (w, p1, p2)s that shifted
def get_changed_rows(word, start_pds, end_pds, models):
    
    rows = []
    
    for start_pd in start_pds:
        for end_pd in end_pds:
            
            # Get word score
            start_model = models[start_pd]
            end_model = models[end_pd]
            score = get_procrustes(word, start_model, end_model)
            
            rows.append([word, start_pd, end_pd, score, 1])
    return rows

# Adds remaining rows for (w, p1, p2)s that didn't shift
def get_unchanged_rows(word, rows, pd_list, models):
    
    change_pds = []
    if len(rows) > 0:
        change_pds = [(row[1], row[2]) for row in rows]
    
    unchanged_rows = []
    
    for s_index, start_pd in enumerate(pd_list):
        
        for end_pd in pd_list[s_index+1:]:
            if (start_pd, end_pd) not in change_pds:
                
                # Get word score
                start_model = models[start_pd]
                end_model = models[end_pd]
                score = get_procrustes(word, start_model, end_model)
                unchanged_rows.append([word, start_pd, end_pd, score, 0])
    
    return unchanged_rows
    
# Returns the Procrustes score for `word` between `start_model` and `end_model`
def get_procrustes(word, start_model, end_model):
    
    try:
        score = ProcrustesAligner(start_model, end_model).get_score(word)
    except KeyError:
        score = None
    
    print("SCORE:", score)
    return score
    
# Train and return a support vector classifier to classify 
# (word, date_range_1, date_range_2) as shifted or not shifted
def train_classifier(train_set): 
    
    x_train = [[row[1][0], row[2][0], row[3]] for row in train_set]
    y_train = [row[-1] for row in train_set]
    
    classifier = SVC(gamma='scale')
    classifier.fit(x_train, y_train)
    
    return classifier

# Predicts and evaluates whether the (word, date_range_1, date_range_2) entries
# in the test set are examples of shift.
# Returns the predicted classes
def test_classifier(classifier, test_set):
    
    test_vectors = [[row[1][0], row[2][0], row[3]] for row in test_set]
    gt = [row[-1] for row in test_set]
    predictions = classifier.predict(test_vectors)
    
    
    # Evaluation
    cm = metrics.confusion_matrix(gt, predictions)
    print(cm, flush=True)
    score = classifier.score(test_vectors, gt)
    print("Accuracy:", score, flush=True)
    f1 = metrics.f1_score(gt, predictions)
    print("F1:", f1)
    scores = metrics.precision_recall_fscore_support(gt, predictions)
    print(scores, flush=True)
    report = metrics.classification_report(gt, predictions, output_dict=True)
    print(report['weighted avg'])
    
    true_zeros = [item for item in gt if item==0]
    true_ones = [item for item in gt if item==1]
    print(len(true_zeros), len(true_ones))
    
    pred_zeros = [item for item in predictions if item==0]
    pred_ones = [item for item in predictions if item==1]
    print(len(pred_zeros), len(pred_ones))
    
    return predictions
        

# Analyzes classifier performance by lexical category
def analyze_by_category(train_set, test_set, classifier):
    
    fpath = "dev_eval_sets/ndhe_final_word_list_with_categories.txt"
    
    file = open(fpath, "r", encoding='ANSI')
    
    
    word_categories = {}
    
    num_adj_and_n = 0
    num_adj_only = 0
    for line in file:
        
        entry = line.split()
        word = entry[0]
        
        if entry[-1] == "N,Adj":
            word_categories[word] = ["N"]
            num_adj_and_n += 1
        elif entry[-1] == "Adj":
            word_categories[word] = ["Adj"]
            num_adj_only += 1
        elif entry[-1] == "V":
            word_categories[word] = ["V"]
        elif entry[-1] == "N,V":
            word_categories[word] = ["N", "V"]
        else:
            word_categories[word] = ["N"]
        
    
    # Get results for each lexical category
    print("Noun Data:")
    train_nouns = [entry for entry in train_set if "N" in word_categories[entry[0]]]
    test_nouns = [entry for entry in test_set if "N" in word_categories[entry[0]]]
    
    print(len(train_nouns), len(test_nouns), "out of", len(train_set), len(test_set))
    
    noun_predictions = test_classifier(classifier, test_nouns)
    
    print("Verb Data:")
    train_verbs = [entry for entry in train_set if "V" in word_categories[entry[0]]]
    test_verbs = [entry for entry in test_set if "V" in word_categories[entry[0]]]
    
    print(len(train_verbs), len(test_verbs), "out of", len(train_set), len(test_set))
    
    verb_predictions = test_classifier(classifier, test_verbs)
    
    print("Combined Noun and Verb Data:")
    test_nouns_verbs = test_nouns + test_verbs
    noun_verb_predictions = test_classifier(classifier, test_nouns_verbs)
    
    print("Adjective Data:")
    train_adjs = [entry for entry in train_set if "Adj" in word_categories[entry[0]]]
    test_adjs = [entry for entry in test_set if "Adj" in word_categories[entry[0]]]
    print(len(train_adjs), len(test_adjs), "out of", len(train_set), len(test_set))
    
    adj_predictions = test_classifier(classifier, test_adjs)
  
    
# Cleans data from the Nuevo Diccionario Historico
def clean_eval_set():
    
    path = "dev_eval_sets/diccionario_historico_completo.txt"
    
    bad_lines = ["NUEVO DICCIONARIO HISTÓRICO DEL ESPAÑOL",
                 "Artículo | Familia",
                 "Cronológico | Frecuencia",
                 "Créditos|Cómo se cita|Presentación|Estructura|Ayuda",
                 "Búsqueda",
                 "ConsultaLimpiar consulta",
                 "Página: 78de84|<<>>|Nº pag.",
                 "Ir"]
    
    words = []
    
    file = open(path, "r", encoding='utf-8')
    for line in file:
        if len(line) > 0 and line not in bad_lines and "Página" not in line and "(" in line:
            word = line[: line.index("(")]
            if "," in word:
                word = word[:word.index(",")]
                words.append(word)

    search_string = ", ".join(words)
    print(search_string)
    
    
# Downloads all Google n-gram files for the parameter `ngram_size`
def write_files(ending, ngram_size):
    
    import requests
    from py.path import local
    
    session = requests.Session()
    
    fname = "googlebooks-spa-all-"+str(ngram_size)+"gram-20120701-" + ending + ".gz"

    url = "http://storage.googleapis.com/books/ngrams/books/" + fname

    output = local("/scratch/network/efleisig/downloads/google_ngrams/5")

    request = session.get(url, stream=True)
    with output.join(fname).open('wb') as f:
        for num, chunk in enumerate(request.iter_content(1024)):
            f.write(chunk)
    


print("Begin semantic shift analysis...", flush=True)
ngram_dt = OrderedDict({(1522, 1899): [], (1900, 1949): [], (1950, 1969): [],
                        (1970, 1984): [], (1985, 1994): [], (1995, 1999): [],
                        (2000, 2004): [], (2005, 2009): []})
unigram_dt = OrderedDict({(1522, 1899): [], (1900, 1949): [], (1950, 1969): [],
                          (1970, 1984): [], (1985, 1994): [], (1995, 1999): [],
                          (2000, 2004): [], (2005, 2009): []})


for ngram_size in range(2, 6):
    
    create_ngram_lists_from_server(unigram_dt, ngram_dt, ngram_size, 1000000)
    
    ngram_dict = pickle.load( open('ngram_dict_' + str(ngram_size) + '_final.pickle', "rb" ) )    
    make_word_embeddings(ngram_dt, ngram_size)
    
    word_mtx_changed, word_mtx_unchanged = get_word_vectors("ndhe_final_word_list.pickle", list(unigram_dt.keys()), ngram_size)
    with open(str(ngram_size) + 'gram_word_mtx_changed.pickle', 'wb') as handle:
        pickle.dump(word_mtx_changed, handle)
    with open(str(ngram_size) + 'gram_word_mtx_unchanged.pickle', 'wb') as handle:
        pickle.dump(word_mtx_unchanged, handle)
        

    # Classifier training and testing
        
    word_mtx_changed = pickle.load( open( str(ngram_size) + 'gram_word_mtx_changed.pickle', "rb" ) )
    word_mtx_unchanged = pickle.load( open( str(ngram_size) + 'gram_word_mtx_unchanged.pickle', "rb" ) )
    
    word_mtx_changed = [row for row in word_mtx_changed if row[-2] != None]
    word_mtx_unchanged = [row for row in word_mtx_unchanged if row[-2] != None]

    with open(str(ngram_size) + 'gram_word_mtx_changed_clean.pickle', 'wb') as handle:
        pickle.dump(word_mtx_changed, handle)
    with open(str(ngram_size) + 'gram_word_mtx_unchanged_clean.pickle', 'wb') as handle:
        pickle.dump(word_mtx_unchanged, handle)
        
    #word_mtx_changed = pickle.load( open( str(ngram_size) + 'gram_word_mtx_changed_clean.pickle', "rb" ) )
    #word_mtx_unchanged = pickle.load( open( str(ngram_size) + 'gram_word_mtx_unchanged_clean.pickle', "rb" ) )    
    
    random.shuffle(word_mtx_changed)
    random.shuffle(word_mtx_unchanged)

    split = int(.8*len(word_mtx_changed))
    word_mtx_unchanged = word_mtx_unchanged[:len(word_mtx_changed)]
    train_set = word_mtx_changed[:split] + word_mtx_unchanged[:split]
    test_set = word_mtx_changed[split:] + word_mtx_unchanged[split:]
    
    random.shuffle(train_set)
    random.shuffle(test_set)
        
    with open(str(ngram_size) + 'gram_train_set.pickle', 'wb') as handle:
        pickle.dump(train_set, handle)
    with open(str(ngram_size) + 'gram_test_set.pickle', 'wb') as handle:
        pickle.dump(test_set, handle)
        
    #train_set = pickle.load( open( str(ngram_size) + 'gram_train_set.pickle', "rb" ) )
    #test_set = pickle.load( open( str(ngram_size) + 'gram_test_set.pickle', "rb" ) )
        
        
    train_zeros = [item for item in train_set if item[-1]==0]
    train_ones = [item for item in train_set if item[-1]==1]
    
    test_zeros = [item for item in test_set if item[-1]==0]
    test_ones = [item for item in test_set if item[-1]==1]
    
    
    classifier = train_classifier(train_set)
    print("------ RESULTS FOR " + str(ngram_size) + "GRAM ------")
    predictions = test_classifier(classifier, test_set)
    
    analyze_by_category(train_set, test_set, classifier)   
    
    
    # Find and visualize examples    
    
    example_list = ["guerra", "lucha", "red", "batalla", "indie", "prensa", 
                    "terrorista", "morboso", "insurgente", "móvil", "capri", 
                    "red", "viral", "cameo", "remezcla", 
                    "indie", "spam", "virtual", "portal", "nube", "enlace", 
                    "descargable", "desarrollador", "buscador", "tableta", 
                    "partisano", "leproso", "apestado", "corrupción", 
                    "propaganda", "coche", "azafata", "terrorista", "apagón", 
                    "égida", "clarín", "balística", "móvil", "sida"]
    
    examples_clean = ["guerra", "lucha", "red", "batalla", "indie", "prensa", 
                     "terrorista", "morboso", "insurgente", "movil", "capri", 
                     "red", "viral", "cameo", "remezcla", 
                     "indie", "spam", "virtual", "portal", "nube", "enlace", 
                     "descargable", "desarrollador", "buscador", "tableta", 
                     "partisano", "leproso", "apestado", "corrupcion", 
                     "propaganda", "coche", "azafata", "terrorista", "apagon", 
                     "egida", "clarin", "balistica", "movil", "sida"]
    
    models = {}
    periods_list = list(unigram_dt.keys())
    for pd in periods_list:
        model_name = "word2vec_" + str(ngram_size) + "gram_"+ str(pd[0]) + "kv.model"
        models[pd] = load_model(model_name)
        
    for index1, p1 in enumerate(periods_list):
        for p2 in periods_list[index1+1:]:
            
            for word in example_list:
                print(word, p1, p2, get_procrustes(word, models[p1], models[p2]), flush=True)
        
    for index, word in enumerate(example_list):
        clean_word = examples_clean[index]

        graph_word_embedding(ngram_dt, word, clean_word, ngram_size)
    
