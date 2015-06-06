# coding: utf-8

# Author: Motaz Saad

import sys
from random import shuffle
##################################################################
#Example Usage:
#python aligning-docs-by-lsi-demo.py ~/corpus/uncorporaorg.0.9.p2.ar ~/corpus/uncorporaorg.0.9.p2.en parallel ~/tmp/docs_aligned_by_lsi/ un ~/tmp/docs_aligned_by_lsi/

def usage():
	print 'Usage: ', sys.argv[0], '<corpus name> <source corpus file> <target corpus file> <corpus type> <model name> <working path>'
##################################################################

if len(sys.argv) < 4: usage(); sys.exit(2)

'''
This software is a demo aligning comparable documents using interlanguage links. The method is described in 

Motaz Saad. Mining Documents and Sentiments in Cross-lingual Context. PhD thesis, Université de Lorraine, January 2015.
https://sites.google.com/site/motazsite/Home/publications/saad_phd.pdf

'''

import imp
tp = imp.load_source('textpro', 'textpro.py')


def main(argv):
	source_test_corpus_file = sys.argv[1]
	target_test_corpus_file = sys.argv[2]
	corpus_type = sys.argv[3]
	model_path = sys.argv[4]
	model_name = sys.argv[5]
	output_path = sys.argv[6]
	
	# aligning documents
	source_test_corpus = tp.load_corpus(source_test_corpus_file, corpus_type)
	target_test_corpus = tp.load_corpus(target_test_corpus_file, corpus_type)
	
	shuffle(target_test_corpus) # shuffle the corpus because it is parallel (already aligned)
	
	tp.align_documents_lsi(source_test_corpus, target_test_corpus, model_path, model_name, output_path)


##################################################################


if __name__ == "__main__":
	main(sys.argv)



