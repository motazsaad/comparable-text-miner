#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
# Comparable text miner

# Description 
Comparable document miner: Arabic-English morphological analysis, text processing, n-gram features extraction, POS tagging, dictionary translation, documents alignment, corpus information, text classification, tf-idf computation, text similarity computation, HTML documents cleaning, and others. 

This code is implemented by Motaz SAAD (motaz.saad@gmail.com) during his PhD work. The PhD thesis is available at: https://sites.google.com/site/motazsite/Home/publications/saad_phd.pdf

Motaz Saad. Mining Documents and Sentiments in Cross-lingual Context. PhD thesis, Université de Lorraine, January 2015.

This code processes Arabic and English text. To use this software, load it as follows:

import imp
tp = imp.load_source('textpro', 'textpro.py')

Then, you can use functions as follows:

clean_text = process_text(text)

# Dependencies
This software depends on the following python packages scipy, numpy, nltk, sklearn, bs4. Please make sure that they are installed before using this software. 

# References
This software uses the following resources:
- Arabic stopwords: http://www.ranks.nl/stopwords/arabic 
- Open Multilingual WordNet (OMW) dictionaries http://compling.hss.ntu.edu.sg/omw/ The references of OMW are listed below:
	- Francis Bond and Kyonghee Paik (2012), A survey of wordnets and their licenses In Proceedings of the 6th Global WordNet Conference (GWC 2012). Matsue. 64–71.
	- Francis Bond and Ryan Foster (2013), Linking and extending an open multilingual wordnet. In 51st Annual Meeting of the Association for Computational Linguistics: ACL-2013. Sofia. 1352–1362. 

- ISRI Arabic Stemmer, which is a rooting algorithm for Arabic text. The reference of ISRI Arabic Stemmer is below:
	- Taghva, K., Elkoury, R., and Coombs, J. 2005. Arabic Stemming without a root dictionary. Information Science Research Institute. University of Nevada, Las Vegas, USA.
 

- This software modifies the ISRI Arabic Stemmer to perform light stemming for Arabic words. 

'''

import sys
import os.path
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.util import ngrams
from nltk.corpus import wordnet as omw # open multilingual wordnet
from nltk.stem.isri import ISRIStemmer

import sqlite3


import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from random import shuffle
from scipy.spatial import distance
import math

from bs4 import BeautifulSoup

import logging
logging.basicConfig(format='%(levelname)s : %(asctime)s : %(message)s', level=logging.INFO)


import re
whiteSpace = re.compile(r'\s+')

#import imp
#tp = imp.load_source('textpro', 'textpro.py')


x_seperator = '\nXXXXXXX\n' # define document separator (7 Xs). This separator is used when all the docs are in one file (a corpus file)

##################################################################

# Arabic diacritics
arabic_punct = ''' ` ÷ × ؛ < > _ ( ) * & ^ % ] [ ـ ، / : " ؟ . , ' { } ~ ¦ + | !  ”  …  “ –   ـ  '''
arabic_diacritics = '''  َ     ُ       ِ      ّ       ً      ٌ       ٍ      ْ     '''

arabic_punctUnicode = arabic_punct.decode('utf-8')
arabic_punct = arabic_punct.split()
arabic_punctUnicode = arabic_punctUnicode.split()

arabic_diacritics_unicode = arabic_diacritics.decode('utf-8')
arabic_diacritics = arabic_diacritics.split()
arabic_diacritics_unicode = arabic_diacritics_unicode.split()

english_punt = list(string.punctuation)
english_puntUnicode = list(string.punctuation.decode('utf-8'))

# Arabic punctuations and dicritis + English and Arabic
punctuations = set( english_punt + english_puntUnicode + arabic_punct + arabic_punctUnicode + arabic_diacritics + arabic_diacritics_unicode)

englishStopWords = stopwords.words('english')
englishStopWords_unicode = ' '.join(englishStopWords).decode('utf-8').split()

# Arabic stopwords. This list are obtained from http://www.ranks.nl/stopwords/arabic 

asw = open('stopwords.txt').read()

aswUinicode = asw.decode('utf-8')
arabicStopWords =  asw.split() + aswUinicode.split()

# Arabic stopwords. This list are obtained from https://code.google.com/p/stop-words/

asw2 = ''

# Arabic and English stopwords
all_stopwords = set(englishStopWords + englishStopWords_unicode + arabicStopWords)



###################################################################################
###################################################################################

# remove punctcutions
def remove_punct(word):
	for c in word: return ''.join(ch for ch in word if not ch in punctuations) # remove punctuation
###################################################################################

# takes a string of text and returns the word list (tonkized words)
# processing includes: removing diacritics and punctcutions, removing stopwords, and tokenizing
def process_text(text, removePunct=True, removeSW=True, removeNum=False):
	text = remove_diacritics(text)# remove arabic diacritics
	word_list = nltk.tokenize.wordpunct_tokenize(text.lower())
	if removePunct:
		word_list = [ w for w in word_list if not w in punctuations ]
		word_list = [ remove_punct(w) for w in word_list ]
	if removeSW: word_list = [ w for w in word_list if not w in all_stopwords ]
	if removeNum: word_list = [ w for w in word_list if not w.isdigit() ]
	word_list = [ w for w in word_list if w]# remove empty words

	return word_list
###################################################################################

# remove arabic diacritics
def remove_diacritics(text):
	arstemmer = ISRIStemmer()
	result = arstemmer.norm(text, num=1) #  remove diacritics which representing Arabic short vowels
	return result
###################################################################################

"""
ISRI Arabic Stemmer

The algorithm for this stemmer is described in:

Taghva, K., Elkoury, R., and Coombs, J. 2005. Arabic Stemming without a root dictionary.
Information Science Research Institute. University of Nevada, Las Vegas, USA.

The Information Science Research Institute’s (ISRI) Arabic stemmer shares many features
with the Khoja stemmer. However, the main difference is that ISRI stemmer does not use root
dictionary. Also, if a root is not found, ISRI stemmer returned normalized form, rather than
returning the original unmodified word.

Additional adjustments were made to improve the algorithm:

1- Adding 60 stop words.
2- Adding the pattern (تفاعيل) to ISRI pattern set.
3- The step 2 in the original algorithm was normalizing all hamza. This step is discarded because it
increases the word ambiguities and changes the original root.

"""


# takes a word list and returns the root for each Arabic words
def getRootAr(word_list):
	result = []
	arstemmer = ISRIStemmer()
	for word in word_list: result.append(arstemmer.stem(word))
	return ' '.join(result)
###################################################################################

# Arabic light stemming for Arabic text
# takes a word list and perform light stemming for each Arabic words
def lightStemAr(word_list):
	result = []
	arstemmer = ISRIStemmer()
	for word in word_list:
		word = arstemmer.norm(word, num=1)  #  remove diacritics which representing Arabic short vowels  
		if not word in arstemmer.stop_words:   # exclude stop words from being processed
			word = arstemmer.pre32(word)        # remove length three and length two prefixes in this order
			word = arstemmer.suf32(word)        # remove length three and length two suffixes in this order
			word = arstemmer.waw(word)          # remove connective ‘و’ if it precedes a word beginning with ‘و’
			word = arstemmer.norm(word, num=2)       # normalize initial hamza to bare alif
		result.append(word)
	return ' '.join(result)

###################################################################################

# combine rooting and light stemming: if light stemming alogrithm manage to reduce word form, then the light stem is returned, else, the root is returned
def arMorph(text_list):
	result = []
	for word in word_list:
		sol = None
		root = getRootAr(word)
		lightStem = lightStemAr(word)
		if t == lightStem: sol = root
		else: sol = lightStem
		result.append(sol)
	return ' '.join(result)

###################################################################################

# execlude stopwords from a list of words
def exclude_stopwords(word_list):
	return [ w for w in word_list if not w in all_stopwords ]

###################################################################################

# return lemma for english text
def getLemma(text, contextFlag=False):
	lemmatizer = WordNetLemmatizer()
	#'NN':wordnet.NOUN,'JJ':wordnet.ADJ,'VB':wordnet.VERB,'RB':wordnet.ADV
	wordnet_tag ={'NN':'n','JJ':'a','VB':'v','RB':'r'}
	result = None
	if text.split() == 1: # on word
		tokenized = word_tokenize(t)
		tagged = pos_tag(tokenized)[0]
		lemma = ''
		try: lemma = lemmatizer.lemmatize(tagged[0],wordnet_tag[tagged[1][:2]])
		except: lemma = lemmatizer.lemmatize(tagged[0])
		result = lemma
	elif text.split() > 1 and contextFlag == True: # mutiple words i.e. text and without considering the context
		resultList = []
		for t in text.split():
			tokenized = word_tokenize(t)
			tagged = pos_tag(tokenized)[0]
			lemma = ''
			try: lemma = lemmatizer.lemmatize(tagged[0],wordnet_tag[tagged[1][:2]])
			except: lemma = lemmatizer.lemmatize(tagged[0])
			resultList.append(lemma)
		result = ' '.join(resultList)
	else: # mutiple words i.e. text and consider the context
		resultList = []
		tokens = word_tokenize(text)
		tagged = pos_tag(tokens)
		for t in tagged:
			try: resultList.append(lemmatizer.lemmatize(t[0],wordnet_tag[t[1][:2]]))
			except: resultList.append(lemmatizer.lemmatize(t[0]))
		result = ' '.join(resultList)
	return result
###################################################################################

# Given a Naive Bayes classifier, classify a text with a given certaintaity
def classify_text(text, classifier, certainity, g, unicodeFlag):
	#1. process text
	if unicodeFlag: text = text.decode('utf-8')
	word_list = process_text(text, removePunct=True, removeSW=False, removeNum=False)

	#2. generate ngrams
	mygrams = generate_ngrams(word_list, g)

	#3. generate features from ngrams
	feats = generate_features(mygrams)

	#4. classify
	probs = classifier.prob_classify(feats)
	label = probs.max()
	if probs.prob(label) >= certainity: return label, probs.prob(label)
	else: return 'none', probs.prob(label)

###################################################################################
# generates n-gram (g = num of grams)
# for example, if g=3, then the fuction will generate unigrams, bigrams, and tri-grams from the text.
def generate_ngrams(word_list, g):
	mygrams = []
	unigrams = [word for word in word_list]
	mygrams += unigrams
	for i in range(2,g+1): mygrams += ngrams(word_list, i)
	return mygrams
###################################################################################

# generate n-gram features in the form (n-gram, True), i.e., binary feature. In other words, the n-gram exists
def generate_features(mygrams):
	feats = dict([(word, True) for word in mygrams])
	return feats
###################################################################################

# generate features for a doc from selected features grams (selected from a corpus)
# taks 2 parameters:
# 1. document feature grams
# 2. corpus selected feature grams
def build_features(doc_feat_grams, corpus_feat_grams):
	doc_grams = set(doc_feat_grams)
	feats = dict([(word, True) for word in doc_grams if word in corpus_feat_grams])
	return feats
###################################################################################

# evaluate predicted results using true values.
# evaluation metrics are acccuracy, precicion, recall and f-measure.
def evaluate(trueValues, predicted, decimals, note):
	print note
	label = 1
	avg = 'weighted'
	a = accuracy_score(trueValues, predicted)
	p = precision_score(trueValues, predicted, pos_label=label, average=avg)
	r = recall_score(trueValues, predicted, pos_label=label, average=avg)
	avg_f1 = f1_score(trueValues, predicted, pos_label=label, average=avg)
	fclasses = f1_score(trueValues, predicted, average=None)
	f1c1 = fclasses[0]; f1c2 = fclasses[1]
	fw = (f1c1 + f1c2)/2.0

	print 'accuracy:\t', str(round(a,decimals))
	print 'precision:\t', str(round(p,decimals))
	print 'recall:\t', str(round(r,decimals))
	print 'avg f1:\t', str(round(avg_f1,decimals))
	print 'c1 f1:\t', str(round(f1c1,decimals))
	print 'c2 f1:\t', str(round(f1c2,decimals))
	print 'avg(c1,c2):\t', str(round(fw,decimals))
	print '------------'

###################################################################################


# split a parallel or comparable corpus into two parts
def split_corpus(source_corpus, target_corpus, percentage):
	print 'len(source_corpus) == len(target_corpus)', len(source_corpus), '==' , len(target_corpus) , len(source_corpus) == len(target_corpus)
	if len(source_corpus) != len(target_corpus): print 'FAILED: the corpus is not aligned correclty'; return None

	size = len(source_corpus)
	p1 = int (len(source_corpus) * percentage )
	p2 = len(source_corpus) - p1
	print 'size, p1, p2: ', size, p1, p2

	udoc = []

	for e,a in zip(source_corpus,target_corpus): udoc.append( (e,a) )

	shuffle(udoc)

	source_p1 = [] ; source_p2 = []
	target_p1 = [] ; target_p2 = []

	for d in udoc[:p1]: source_p1.append( d[0] )


	for d in udoc[:p1]: target_p1.append( d[1] )


	for d in udoc[p1:]: source_p2.append( d[0] )


	for d in udoc[p1:]: target_p2.append( d[1] )

	return source_p1, target_p1, source_p2, target_p2

##################################################################################
##################################################################################
##################################################################################



# load WordNet (WN) dictionaries
# Dictionaries are obtained from Open Multilingual WordNet website: http://compling.hss.ntu.edu.sg/omw/

# To cite these dictionaries:
# Francis Bond and Kyonghee Paik (2012), A survey of wordnets and their licenses In Proceedings of the 6th Global WordNet Conference (GWC 2012). Matsue. 64–71.
# Francis Bond and Ryan Foster (2013), Linking and extending an open multilingual wordnet. In 51st Annual Meeting of the Association for Computational Linguistics: ACL-2013. Sofia. 1352–1362. 

eng_dict_file = 'wordnet/wn-data-eng.tab'
arb_dict_file = 'wordnet/wn-data-arb.tab'

eng_dict_lines = open(eng_dict_file).readlines()
arb_dict_lines = open(arb_dict_file).readlines()

eng_dict_key = []; eng_dict_word = [];
arb_dict_key = []; arb_dict_word = [];

for l in eng_dict_lines:
	tokens = l.split('\t')
	key = tokens[0][:-2].strip()
	eng_dict_key.append(key)
	word = tokens[2].strip().decode('utf-8')
	eng_dict_word.append(word)

for l in arb_dict_lines:
	tokens = l.split('\t')
	key = tokens[0][:-2].strip()
	arb_dict_key.append(key)
	word = tokens[2].strip().decode('utf-8')
	arb_dict_word.append(word)

###################################################################################
# translation functions using WN bilingual dictionaries
def translate_en2ar(word):
	translations = []
	keys = []
	for i in range(len(eng_dict_word)):
		if word == eng_dict_word[i]: keys.append(eng_dict_key[i])

	for i in range(len(arb_dict_key)):
		for j in range(len(keys)):
			if keys[j] == arb_dict_key[i]: translations.append(arb_dict_word[i])

	return set(translations)
###################################################################################
def translate_ar2en(word):
	translations = []
	keys = []

	for i in range(len(arb_dict_word)):
		if word == arb_dict_word[i]: keys.append(arb_dict_key[i])

	for i in range(len(eng_dict_key)):
		for j in range(len(keys)):
			if keys[j] == eng_dict_key[i]: translations.append(eng_dict_word[i])

	return set(translations)
##################################################################################
##################################################################################
##################################################################################

# binary similarity between two binary vectors
def sim_bin(s_vector,t_vector): return 1 - distance.jaccard(s_vector, t_vector)

# cosine similarity between two wieghted vectors
def sim_cosine(s_vector,t_vector): return 1 - distance.cosine(s_vector, t_vector)

##################################################################################
##################################################################################
##################################################################################

# computes tfidf wieghts for words in a given document. The function needs the corpus to compute idf
def tf_idf(word, document, corpus):
	base = 10
	corpus_size = float(len(corpus))
	tf =  document.count(word)
	doc_freq = float ( sum(1 for doc in corpus if word in doc) )
	idf = math.log( (corpus_size /  doc_freq ), base )
	tf_idf = tf * idf

	return tf_idf
##################################################################################


##################################################################################
# Compute average number of sentences per document for a corpus collectection

def avgSenPerArticle(corpus):
	avg = 0.0
	for d in corpus:
		n = d.splitlines()
		avg += n
	avg /= len(corpus)
	return avg

##################################################################################

##################################################################################
# Compute average number of words per document for a corpus collectection

def avgWordsPerArticle(corpus):
	avg = 0.0
	for d in corpus:
		n = len(d.split())
		avg += n
	avg /= len(corpus)
	return avg

##################################################################################

##################################################################################
# Compute vocabulary size for a text
def vocab(text):
	tok = text.split()
	v = set(tok)
	return len(v)

##################################################################################

##################################################################################

# remove empty lines and white spaces (remove empty lines and keep '\n' in the text)
def pretty_print(text):
	lines = text.splitlines()
	filtered1 = filter(lambda x: not re.match(r'^\s*$', x), lines)
	filtered2 = [whiteSpace.sub(' ', l).strip() for l in filtered1]
	cleantext = '\n'.join(filtered2)
	return cleantext
##################################################################################

# clean html tages from a text
def strip_html_tags(text):
    soup = BeautifulSoup(text)
    doc = pretty_print(soup.get_text())

    return doc
##################################################################################

# find text between two substrings
def find_between(text , first, last ):
	try:
		start = text.index( first ) + len( first )
		end = text.index( last, start )
		return text[start:end]
	except ValueError:
		return None
##################################################################################

def prepare_gensim_corpus(corpus_name, corpus_path, corpus_type, language, min_freq=5, doc_separator=x_seperator):
	
	print 'building gensim corpus and dictionary for', corpusName, lang, 'corpora'
	corpus_file = corpus_path + corpus_name
	train = None
	if corpus_type == 'comparable':
		train = open(corpus_file).read().decode('utf-8').split(doc_separator); del train[-1]
	if corpus_type == 'parallel':
		train = open(corpus_file).read().decode('utf-8').splitlines()
	if not train: print 'corpus type is not supported... The corpus should be parallel or comparable'; return None;
	
	print 'loading corpus'
	texts = [[word for word in tp.process_text(document, removePunct=True, removeSW=True, removeNum=True)] for document in train]
	print 'tokenizing'
	all_tokens = [item for sublist in texts1 for item in sublist]
	print 'mark tokens which have frequency less than', min_freq
	tokens_once = set([k for k, v in Counter(all_tokens).iteritems() if v < min_freq ])
	print '|D|=' , len(texts)
	print 'filter low frequency tokens'
	texts = [[word for word in text if word not in tokens_once] for text in texts]
	print '|D|=' , len(texts)
	print 'building dictionary'
	dictionary = corpora.Dictionary(texts)
	print 'saving dictionary'
	dictFile = corpus_file + corpusName + '.dict'
	dictionary.save(dictFile) 
	print 'building corpus in  mm format'
	corpus = [dictionary.doc2bow(text) for text in texts]
	print 'saving corpus'
	corpusFile = corpus_file + corpusName + '.mm'
	corpora.MmCorpus.serialize(corpusFile, corpus)
	print 'computing tfidf'
	tfidf = models.TfidfModel(corpus) # tfidf model 
	corpus_tfidf = tfidf[corpus] # tfidf corpus 
	print 'saving tfidf corpus'
	corpus_tfidf_file = corpus_file + corpusName + '.tfidf.mm'
	corpora.MmCorpus.serialize(corpus_tfidf_file, corpus_tfidf)
	print 'gensim corpus is ready'
##################################################################################
	
def build_lsi_model(corpus_name, corpus_path, topics=300):
	print 'building lsi model for', corpus_name, 'corpus'
	dictFile = corpus_file + corpus_name + '.dict'
	corpus_tfidf_file = corpus_file + corpus_name + '.tfidf.mm'
	
	print 'loading dictionary ...'
	dictionary = corpora.Dictionary.load(dictFile)
	print 'loading tfidf corpus ...'
	corpus_tfidf = corpora.MmCorpus(corpus_tfidf_file)
	print 'building lsi model'
	lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=topics)
	print 'saving lsi'
	lsiFile = corpus_file + corpus_name + '.lsi'
	lsi.save(lsiFile)
	print 'lsi model is ready'
##################################################################################

def align_documents_lsi(corpus_path, source_corpus_name, target_corpus_name, source_language, target_language, top_n, model_path, model_name, output_path, doc_separator=x_seperator):
	print 'aligning', source_corpus_name, 'with', target_corpus_name, 'using LSI'
	
	dictionaryFile = model_path +  model_name + '.dict'
	lsiFile = model_path +  model_name + '.lsi'
	
	
	dictionary = corpora.Dictionary.load(dictionaryFile) ; print 'dictionary loaded'
	lsi = models.LsiModel.load(lsiFile) ; print 'lsi model loaded'
		
	source_corpus = open(corpus_path + source_corpus_name).read().decode('utf-8').split(doc_separator) 
	target_corpus = open(corpus_path + target_corpus_name).read().decode('utf-8').split(doc_separator) 
	print 'source and target corpora loaded'
	del source_corpus[-1] ; del target_corpus[-1]
	print '# of source docs', len(source_corpus), '\t# of target docs', len(target_corpus)
	
	source_lsi_corpus = generateLSIvectors(source_corpus, dictionary, lsi); print 'projects source_corpus into LSI space'
	target_lsi_corpus = generateLSIvectors(target_corpus, dictionary, lsi); print 'projects target_corpus into LSI space'	
	
	allSims = [] ; doc_tuple = [] ; source_index = 0 
	
	
	for d in source_lsi_corpus:
		target_index, sim = getComparable(d, target_lsi_corpus)
		allSims.append(sim)
		source_doc = source_corpus[source_index] ; target_doc = target_corpus[target_index]
		del target_lsi_corpus[target_index] ; del target_corpus[target_index] # remove the already aligned document from the target corpus
		doc_tuple.append((source_index,target_index, source_doc, target_doc))
		if not target_lsi_corpus: break # all target docs are aligned
		source_index+=1
		
	sortedAllSims = sorted(enumerate(allSims), key=lambda item: -item[1])
	topNList = sortedAllSims[:top_n]
	out = open (output + 'results.txt', 'w')
	count = 0
	print '\n#, src, target, sim'
	for e in topNList:
		i, sim = e
		srcIndx = doc_tuple[i][0] ; targetIndx = doc_tuple[i][1] ; sdoc = doc_tuple[i][2] ; tdoc = doc_tuple[i][3]
		print count, srcIndx, targetIndx, '%0.2f' % sim
		print>>out, count, srcIndx, targetIndx, '%0.2f' % sim
		source_out = open(output_path + str(count) + '.' + source_language, 'w')
		targetout = open(output_path + str(count) + '.' + target_language, 'w')
		print>>srcout, sdoc.encode('utf-8')
		print>>targetout, tdoc.encode('utf-8')
		srcout.close(); targetout.close(); count+=1	
	out.close();
	print 'aligning', source_corpus_name, 'with', target_corpus_name, 'using LSI is done!'
##################################################################################

def align_sentences_lsi(source_doc, target_doc, model_path, model_name, doc_separator=x_seperator):
	print 'Sentence level alignment using LSI'
	
	dictionaryFile = model_path +  model_name + '.dict'
	lsiFile = model_path +  model_name + '.lsi'

	dictionary = corpora.Dictionary.load(dictionaryFile) ; print 'dictionary loaded'
	lsi = models.LsiModel.load(lsiFile) ; print 'lsi model loaded'
		
	source_sentences = source_doc.splitlines() ; target_sentences = target_doc.splitlines()
	
	source_lsi_sentences = generateLSIvectors(source_sentences, dictionary, lsi); print 'projects source_sentences into LSI space'
	target_lsi_sentences = generateLSIvectors(target_sentences, dictionary, lsi); print 'projects target_sentences into LSI space'	
	
	source_index = 0 	
	new_source_doc = [] ; new_target_doc = []
	
	for d in source_lsi_sentences:
		target_index, sim = getComparable(d, target_sentences)
		allSims.append(sim)
		source_sent = source_sentences[source_index] ; target_sent = target_sentences[target_index]
		del target_lsi_sentences[target_index] ; del target_sentences[target_index] # remove the already aligned sentences from the target document
		new_source_doc.append(source_sent) 
		new_target_doc.append(target_sent)
		if not target_lsi_sentences: break # all target sentences are aligned
		source_index+=1
		
	return new_source_doc, new_target_doc
##################################################################################
# projecting a corpus into LSI space
def generateLSIvectors(corpus, dictionary, lsi):
	LSIcorpus = []
	for d in corpus:		
		vec_bow = dictionary.doc2bow(tp.process_text(d))
		vec_lsi = lsi[vec_bow] 
		LSIcorpus.append(vec_lsi)
	return LSIcorpus
##################################################################################
# given a source doc, get the most comparable document from the target corpus
# returns the index of the target document in the the target corpus
def getComparable(source_lsi_doc, target_lsi_corpus):
	sims = []
	for i in range(len(target_lsi_corpus)):
		sims.append( matutils.cossim(source_lsi_doc, target_lsi_corpus[i]) )
	sortedSims = sorted(enumerate(sims), key=lambda item: -item[1])
	topIndex = sortedSims[0][0]
	topSim = sortedSims[0][1]
	return sortedSims[0]

##################################################################################


##################################################################################
# takses wiki text and a list of language codes, and returns the interlanguage links
# language code list:
# ar arabic
# en english
# fr french
# es Español
# de Deutsch
# it Italiano
# pt portuguese
# fa farsi
# ur urdo
# he hebrew
# ps peshto (Afghānī)
# sd Sindhi (sindi)
# ug Uyghur أويغورية
# pnb punjabi (Pakistan - India)
# ckb kurdi
# arz egyptian
# select * from langlinks where ll_lang = "ar" or ll_lang= "en" or ll_lang= "fr" or ll_lang= "fa" or ll_lang= "he" or ll_lang= "arz" or ll_lang= "ur" or ll_lang= "es" or ll_lang= "it" or ll_lang= "de";

'''
select 
    langlinks_arwiki.ll_title, langlinks_arzwiki.ll_title
from
    langlinks_arwiki
        INNER JOIN
    langlinks_arzwiki
ON
    langlinks_arwiki.ll_from = langlinks_arzwiki.ll_from
where 
	langlinks_arwiki.ll_lang = 'arz' 
	and
	langlinks_arzwiki.ll_lang = 'ar' 
;

CREATE TABLE my_wiki_schema_short.langlinks_arwiki_short AS
SELECT *
FROM my_wiki_schema.langlinks_arwiki
where
    ll_lang = 'en'
    or ll_lang = 'fr'
    or ll_lang = 'fa'
    or ll_lang = 'he'
    or ll_lang = 'arz'
    or ll_lang = 'ur'
    or ll_lang = 'es'
    or ll_lang = 'it'
    or ll_lang = 'de'
	or ll_lang = 'pt'
	or ll_lang = 'ps'
	or ll_lang = 'sd'
	or ll_lang = 'ug'
	or ll_lang = 'pnb'
	or ll_lang = 'ckb'
	or ll_lang = 'simple'
;

'''
#$ gem install sequel mysql sqlite3
#$ sequel mysql://user:password@host/database -C sqlite://interlanguage-links-4-2015.sqlite

# sequel mysql://wiki:wiki@localhost/my_wiki_schema_get_interlanguage_links_sqlshort -C sqlite://interlanguage-links-4-2015.sqlite

lang_list = ['ar', 'en', 'fr', 'es', 'it', 'de', 'fa', 'he', 'ur', 'ps', 'sd', 'ug', 'pnb', 'ckb', 'arz']

def get_interlanguage_links(wiki_text, language_code_list=lang_list):
	interlinks = []
	for code in language_code_list:
		link = find_between(wiki_text, '[[' + code + ':', ']]')
		if link: interlinks.append('[[' + code + ':' + link + ']]')
	return interlinks	
##################################################################################
def get_interlanguage_links_sql(doc_id, db_file):
	interlinks = []
	db = sqlite3.connect(db_file)
	cur = db.cursor()
	sql = '''
	SELECT langlinks_arwiki_short.ll_lang, langlinks_arwiki_short.ll_title
	FROM langlinks_arwiki_short
	where
    	langlinks_arwiki_short.ll_from = '%d' ''' % doc_id
	cur.execute(sql)
	results = cur.fetchall()
	for row in results: 
		lang = row[0] ; title = row[1]
		interlinks.append('[[' + lang + ':' + title + ']]')
	db.close()
	return interlinks
##################################################################################

def load_interlanguage_links(wiki_doc):
	links = find_between(wiki_doc , '<interlanguage_links>', '</interlanguage_links>' )
	return links

##################################################################################

def get_title_from_interlanguage_links(links, language):
	title = find_between(links, '[[' + language + ':' , ']]')
	return title

##################################################################################

def aligning_documents_by_interlanguage_links(source_corpus_file, target_corpus_file, source_language, target_language, output_path):
	logging.info( 'aliging %s and %s wikipeida documents using interlanguage links',  source_language, target_language)
	source_docs = split_wikipedia_docs_into_array(source_corpus_file)
	logging.info( 'source corpus is loaded')
	target_docs = split_wikipedia_docs_into_array(target_corpus_file)
	logging.info( 'target corpus is loaded')
	
	logging.info( 'start aligning...')
	source_out = open(output_path +  source_language + '.wiki.txt', 'w') 
	target_out = open(output_path +  target_language + '.wiki.txt', 'w') 
	
	for i in range(len(source_docs)):
		source_title = get_title_from_interlanguage_links(source_docs[i], source_language)
		for j in range(len(target_docs)):
			target_title = get_title_from_interlanguage_links(target_docs[j], target_language)
			if source_title == target_title:
				text_out = source_docs[i] + '\n' + x_seperator
				print>>source_out, text_out.encode('utf-8')
				text_out = target_docs[j] + '\n' + x_seperator
				print>>target_out, text_out.encode('utf-8')
			
	logging.info( 'aliging by document interlanguage links is done!')
##################################################################################

# takes a wikipedia corpus (extracted by WikiExtractor.py) and splits the corpus into documents and clean them 
def split_wikipedia_docs(corpus_file, output_path, doc_len=30):
	corpus = open(corpus_file).read().split('</doc>')
	logging.info( 'processing %d wikipedia documents...', len(corpus))
	count = 1
	for d in corpus:
		doc = strip_html_tags(d)
		if len(doc.split()) > doc_len: # if the number of words in the document is greater than doc_len, then the document will be extracted
			out = open(output_path + os.path.basename(corpus_file) + str('-%07d' % count) + '.txt', 'w')
			print>>out, doc.encode('utf-8')
			out.close(); count+=1
	logging.info('%d documents are extracted', count)
##################################################################################


# takes a wikipedia corpus (extracted by WikiExtractor.py) and splits the corpus into documents and clean them and returns an array
def split_wikipedia_docs_into_array(corpus_file, doc_len=30):
	documents = []
	corpus = open(corpus_file).read().decode('utf-8').split('</doc>')
	for d in corpus:
		doc = strip_html_tags(d)
		if len(doc.split()) > doc_len: # if the number of words in the document is greater than doc_len, then the document will be extracted
			documents.append(doc)
			
	return 	documents
##################################################################################




##################################################################################
# TODO: group words according to their synset IDs
def omw_syn(word):
	syn = omw.synsets(word, language)[0]
	return syn.lemma_names(lang=language)
##################################################################################

##################################################################################

##################################################################################


##################################################################################





