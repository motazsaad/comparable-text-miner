# Comparable text miner

# Description 
Comparable document miner: Arabic-English morphological analysis, text processing, n-gram features extraction, POS tagging, dictionary translation, documents alignment, corpus information, text classification, tf-idf computation, text similarity computation, HTML documents cleaning 

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


This code processes Arabic and English text. To use this software, load it as follows:

import imp
tp = imp.load_source('textpro', 'textpro.py')

You have to change the path to refer to the place of the file in your machine. Then, you can use functions as follows:

clean_text = process_text(text)



# Dependencies
This software depends on the following python packages scipy, numpy, nltk, sklearn, bs4. Please make sure that they are installed before using this software. 

