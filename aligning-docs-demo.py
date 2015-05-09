# coding: utf-8


# Author: Motaz Saad

import sys
##################################################################
#Example Usage:
#python aligning-docs-demo.py test-text-files/test-in.ar.txt test-text-files/test-out.ar.txt

def usage():
	print 'Usage: ', sys.argv[0], '<corpus path> <source corpus name>, <target corpus name> <source language> <target language> <top n> <lsi model path> <model name> <output path>'
	print 'python arabic-morphological-analysis-demo.py test-text-files/test-in.ar.txt test-text-files/test-out.ar.txt'
##################################################################

#if len(sys.argv) < 10: usage(); sys.exit(2)


'''
This software is a demo aligning Arabic-English comparable documents using LSI model. The method is described in 

https://sites.google.com/site/motazsite/Home/publications/saad_phd.pdf

Motaz Saad. Mining Documents and Sentiments in Cross-lingual Context. PhD thesis, Université de Lorraine, January 2015.

'''

import imp
tp = imp.load_source('textpro', 'textpro.py')

#corpus_path, source_corpus_name, target_corpus_name, source_language, target_language, top_n , model_path, model_name, output_path


def main(argv):
#	inputfile = sys.argv[1]
#	outputfile = sys.argv[2]
#	text = open(inputfile).read().decode('utf-8')
#	word_list = tp.process_text(text) # remove diacritics and punctcutions, stopwords, and tokenize text
#	
#	output_text = 'apply rooting\n' + roots + '\n\n=================\napply light stemming\n' + lightStems
#	output = open(outputfile, 'w')
#	print>>output, output_text.encode('utf-8')
#	output.close()

	# 1st step: training 
	
	prepare_gensim_corpus(corpus_name, corpus_path, corpus_type, language):


##################################################################


if __name__ == "__main__":
	main(sys.argv)



