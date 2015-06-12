# coding: utf-8


# Author: Motaz Saad

import sys
from multiprocessing import Process
from joblib import Parallel, delayed
import logging
logging.basicConfig(format='%(levelname)s : %(asctime)s : %(message)s', level=logging.INFO)

x_seperator = '\nXXXXXXX\n' # define document separator (7 Xs). This separator is used when all the docs are in one file (a corpus file)

##################################################################
#Example Usage:
#python aligning-docs-by-interlinks-demo.py ~/wikipedia/processed/arwiki-20150311-pages-articles.txt ~/wikipedia/processed/arzwiki-20150329-pages-articles.txt ar arz ../docs_aligned_by_links/

def usage():
	print 'Usage: ', sys.argv[0], '<source corpus file> <target corpus file> <source language> <target language> <output path>'

##################################################################

if len(sys.argv) < 6: usage(); sys.exit(2)


'''
This software is a demo aligning wikipeida comparable documents using interlanguage links. The method is described in 

https://sites.google.com/site/motazsite/Home/publications/saad_phd.pdf

Motaz Saad. Mining Documents and Sentiments in Cross-lingual Context. PhD thesis, Université de Lorraine, January 2015.

'''

import imp
tp = imp.load_source('textpro', 'textpro.py')


def main(argv):
	source_corpus_file = sys.argv[1]
	target_corpus_file = sys.argv[2]
	source_language = sys.argv[3]
	target_language = sys.argv[4]
	output_path = sys.argv[5]
	
	if not output_path.endswith('/'): output_path = output_path + '/'
	tp.check_dir(output_path) # if directory does not exist, then create
	
	logging.info( 'aligning %s and %s wikipeida documents using interlanguage links',  source_language, target_language)
	source_docs = tp.split_wikipedia_docs_into_array(source_corpus_file)
	logging.info( 'source corpus is loaded')
	target_docs = tp.split_wikipedia_docs_into_array(target_corpus_file)
	logging.info( 'target corpus is loaded ... start aligning ...')
	
	aligned_corpus = Parallel(n_jobs=3,verbose=100)(delayed(tp.aligning_doc_by_interlanguage_links)(d, target_docs, source_language, target_language, output_path) for d in source_docs)
	
	
	source_out = open(output_path +  source_language + '.wiki.txt', 'w') 
	target_out = open(output_path +  target_language + '.wiki.txt', 'w')
	
	for doc_pair in aligned_corpus:
		if doc_pair[0]: # if not None 
			text_out = doc_pair[0]
			print>>source_out, text_out.encode('utf-8')
			text_out = doc_pair[1]
			print>>target_out, text_out.encode('utf-8')
	
	

##################################################################


if __name__ == "__main__":
	main(sys.argv)



