# Comparable text miner

# Description 
Comparable document miner: Arabic-English morphological analysis, text processing, n-gram features extraction, POS tagging, dictionary translation, documents alignment, corpus information, text classification, tf-idf computation, text similarity computation, HTML documents cleaning 

This code is implemented by Motaz SAAD (motaz.saad@gmail.com) during my PhD work. My PhD thesis is available at: https://sites.google.com/site/motazsite/Home/publications/saad_phd.pdf

Motaz Saad. Mining Documents and Sentiments in Cross-lingual Context. PhD thesis, Université de Lorraine, January 2015.

This code processes Arabic and English text. To use this software, load it as follows:

import imp
tp = imp.load_source('textpro', 'textpro.py')

Then, you can use functions as follows:

clean_text = process_text(text)



# Dependencies
This software depends on the following python packages scipy, numpy, nltk, sklearn, bs4. Please make sure that they are installed before using this software. 

