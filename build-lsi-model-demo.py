# coding: utf-8


# Author: Motaz Saad

import sys
##################################################################
#Example Usage:
#python build-lsi-model-demo.py un ~/corpus/uncorporaorg.0.9.p1.ar ~/corpus/uncorporaorg.0.9.p1.en parallel ~/tmp/docs_aligned_by_lsi/


def usage():
	print 'Usage: ', sys.argv[0], '<corpus name> <source corpus file> <target corpus file> <corpus type> <output path>'
##################################################################

if len(sys.argv) < 4: usage(); sys.exit(2)


'''
This software is a demo of building LSI model for a parallel/comparable corpus. The model can be used to align comparable documents (aligning demo is in a separate program). The method is described in 

Motaz Saad. Mining Documents and Sentiments in Cross-lingual Context. PhD thesis, Université de Lorraine, January 2015.
https://sites.google.com/site/motazsite/Home/publications/saad_phd.pdf

'''

import imp
tp = imp.load_source('textpro', 'textpro.py')


def main(argv):
	corpus_name = sys.argv[1]
	source_corpus_file = sys.argv[2]
	target_corpus_file = sys.argv[3]
	corpus_type = sys.argv[4]
	output_path = sys.argv[5]
	
	# train the LSI model
	# prepare corpus
	source_corpus = tp.load_corpus(source_corpus_file, corpus_type)
	target_corpus = tp.load_corpus(target_corpus_file, corpus_type)
	
	merged_corpus = tp.merge_source_target_docs(source_corpus, target_corpus)
	tp.prepare_gensim_corpus(corpus_name, merged_corpus, output_path)
	
	# build lsi model 
	tp.build_lsi_model(corpus_name, output_path, topics=300)


##################################################################


if __name__ == "__main__":
	main(sys.argv)



