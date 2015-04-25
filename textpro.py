# coding: utf-8


'''
This code is implemented by Motaz SAAD (motaz.saad@gmail.com) during my PhD work. My PhD thesis is available at:

https://sites.google.com/site/motazsite/Home/publications/saad_phd.pdf

Motaz Saad. Mining Documents and Sentiments in Cross-lingual Context. PhD thesis, Universite de Lorraine, January 2015.

@phdthesis{Saad2015phd,
  author = {Motaz Saad},
  title = {{Mining Documents and Sentiments in Cross-lingual Context}},
  school = {Université de Lorraine},
  year = {2015},
  Month = {January}
}


This code processes Arabic and English text.




To use this code, load it as follows:

import imp
tp = imp.load_source('textpro', '/home/motaz/Dropbox/phd/scripts/python/text_processing/textpro.py')

You have to change the path to refer to the place of the file in your machine. Then, you can use functions as follows:

clean_text = process_text(text)



'''

import sys
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
import nltk
from nltk.util import ngrams

from nltk.stem.isri import ISRIStemmer
import nltk


import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from random import shuffle
from scipy.spatial import distance
import math

from bs4 import BeautifulSoup



import re
whiteSpace = re.compile(r'\s+')

#import imp
#tp = imp.load_source('textpro', '/home/motaz/Dropbox/phd/scripts/python/text_processing/textpro.py')


##################################################################

arabic_punct = ''' ` ÷ × ؛ < > _ ( ) * & ^ % ] [ ـ ، / : " ؟ . , ' { } ~ ¦ + | !  ”  …  “ –   ـ  '''
arabic_dicritis = '''  َ     ُ       ِ      ّ       ً      ٌ       ٍ      ْ     '''

arabic_punctUnicode = arabic_punct.decode('utf-8')
arabic_punct = arabic_punct.split()
arabic_punctUnicode = arabic_punctUnicode.split()

arabic_dicritis_unicode = arabic_dicritis.decode('utf-8')
arabic_dicritis = arabic_dicritis.split()
arabic_dicritis_unicode = arabic_dicritis_unicode.split()

english_punt = list(string.punctuation)
english_puntUnicode = list(string.punctuation.decode('utf-8'))

# Arabic punctuations and dicritis + English and Arabic
punctuations = set( english_punt + english_puntUnicode + arabic_punct + arabic_punctUnicode + arabic_dicritis + arabic_dicritis_unicode)

englishStopWords = stopwords.words('english')
englishStopWords_unicode = ' '.join(englishStopWords).decode('utf-8').split()

# Arabic stopwords. This list are obtained from http://www.ranks.nl/stopwords/arabic 

asw = 'أنفسنا مثل حيث ذلك بشكل لدى عن لنا إلى فقط الذي الذى لم او أكثر اي أنا أنت له اذا كيف منها اكثر أي أن وكان وفي سوف حين نفسها هكذا قبل أو حول هنا على لكن فيه عليه قليل صباحا لهم بان يكون خارج هناك مع فوق ما لا هذه و فيها نفسه دون ل آخر ثانية انه من جدا إضافي به بن بعض بها هم أيضا كانت هي لها نحن تم أنفسهم ينبغي وقالت عليها فان لماذا عند وجود الى غير قد عندما مرة هؤلاء إذا كل يمكن أنفسكم فعل ثم لي الآن فى في ديك هذا لن ايضا الذين ليس التى التي وان بعد حتى ان معظم وقد يجري كلا لك ضد انها كان لديه ولا بين خلال وقال بعيدا كما نفسي نحو هو نفسك ولم وهي لقاء وكانت بسبب حاليا ومن الا اما وهو تحت'
aswUinicode = asw.decode('utf-8')
arabicStopWords =  asw.split() + aswUinicode.split()

# Arabic stopwords. This list are obtained from https://code.google.com/p/stop-words/

asw2 = ''

# Arabic and English stopwords
all_stopwords = set(englishStopWords + englishStopWords_unicode + arabicStopWords)



###################################################################################
###################################################################################

# remove punctcutions
def rm_punct(word):
	for c in word: return ''.join(ch for ch in word if not ch in punctuations) # remove punctuation
###################################################################################

# return the word list (tonkized words)
def process_text(text, removePunct=True, removeSW=True, removeNum=False):
	text = remove_diacritics(text)# remove arabic diacritics
	word_list = nltk.tokenize.wordpunct_tokenize(text.lower())
	if removePunct:
		word_list = [ w for w in word_list if not w in punctuations ]
		word_list = [ rm_punct(w) for w in word_list ]
	if removeSW: word_list = [ w for w in word_list if not w in all_stopwords ]
	if removeNum: word_list = [ w for w in word_list if not w.isdigit() ]
	word_list = [ w for w in word_list if w]# remove empty words

	return word_list
###################################################################################

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


# return the root for arabic text
def getRootAr(text):
	result = None
	arstemmer = ISRIStemmer()
	if text.split() == 1: result =  arstemmer.stem(text) # one word
	elif text.split() > 1: # mutiple words i.e. text
		resultList = []
		tok = text.split()
		for t in tok: resultList.append(arstemmer.stem(t))
		result = ' '.join(resultList)
	return result
###################################################################################

# Arabic light stemming for Arabic text
def lightStemAr(text):
	arstemmer = ISRIStemmer()
	result = None
	if text.split() == 1: # one word
		arstemmer.stm = text
		arstemmer.norm(1)       #  remove diacritics which representing Arabic short vowels
		if not arstemmer.stm in arstemmer.stop_words:   # exclude stop words from being processed
			arstemmer.pre32()        # remove length three and length two prefixes in this order
			arstemmer.suf32()        # remove length three and length two suffixes in this order
			arstemmer.waw()          # remove connective ‘و’ if it precedes a word beginning with ‘و’
			arstemmer.norm(2)       # normalize initial hamza to bare alif
		result = arstemmer.stm
	elif text.split() > 1: # mutiple words i.e. text
		resultList = []
		tok = text.split()
		for t in tok:
			arstemmer.stm = t
			arstemmer.norm(1)       #  remove diacritics which representing Arabic short vowels
			if not arstemmer.stm in arstemmer.stop_words:   # exclude stop words from being processed
				arstemmer.pre32()        # remove length three and length two prefixes in this order
				arstemmer.suf32()        # remove length three and length two suffixes in this order
				arstemmer.waw()          # remove connective ‘و’ if it precedes a word beginning with ‘و’
				arstemmer.norm(2)       # normalize initial hamza to bare alif
			resultList.append(arstemmer.stm)
		result = ' '.join(resultList)
	return result

###################################################################################

# combine rooting and light stemming: if light stemming alogrithm manage to reduce word form, then the light stem is returned, else, the root is returned
def arMorph(text):
	result = None
	if text.split() == 1: # the text is one word
		root = getRootAr(text)
		lightStem = lightStemAr(text)
		if t == lightStem: result = root
		else: result = lightStem
	elif text.split() > 1: # the text is mutiple words
		resultList = []
		tok = text.split()
		for t in tok:
			sol = None
			root = getRootAr(t)
			lightStem = lightStemAr(t)
			if t == lightStem: sol = root
			else: sol = lightStem
			resultList.append(sol)
		result = ' '.join(resultList)
	return result

###################################################################################

# remove arabic diacritics
def remove_diacritics(text):
	arstemmer = ISRIStemmer()
	result = None
	arstemmer.stm = text
	arstemmer.norm(1)       #  remove diacritics which representing Arabic short vowels
	result = arstemmer.stm
	return result

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
# These dictionaries are available at:

# cite:

# You have to change the path to refer to the place of the files in your machine.

eng_dict_file = '/home/motaz/Dropbox/phd/dictionaries/omw/eng/wn-data-eng.tab'
arb_dict_file = '/home/motaz/Dropbox/phd/dictionaries/omw/arb/wn-nodia-arb.tab'


eng_dict_lines = open(eng_dict_file).readlines()
arb_dict_lines = open(arb_dict_file).readlines()



eng_dict_key = []; eng_dict_word = [];
arb_dict_key = []; arb_dict_word = [];


for l in eng_dict_lines:
	tokens = l.split()
	key = tokens[0][:-2]
	eng_dict_key.append(key)
	word = tokens[2].decode('utf-8')
	eng_dict_word.append(word)


for l in arb_dict_lines:
	tokens = l.split()
	key = tokens[0][:-2]
	arb_dict_key.append(key)
	word = tokens[2].decode('utf-8')
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
			if keys[j] == arb_dict_key[i]:
				translations.append(arb_dict_word[i])

	return set(translations)

###################################################################################


def translate_ar2en(word):
	translations = []
	keys = []

	for i in range(len(arb_dict_word)):
		if word == arb_dict_word[i]: keys.append(arb_dict_key[i])

	for i in range(len(eng_dict_key)):
		for j in range(len(keys)):
			if keys[j] == eng_dict_key[i]:
				translations.append(eng_dict_word[i])

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
		return ""
##################################################################################
