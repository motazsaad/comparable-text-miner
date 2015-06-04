# coding: utf-8


# Author: Motaz Saad

import sys
##################################################################
#Example Usage:
#python aligning-docs-by-lsi-demo.py ~/wikipedia/processed/arwiki-20150311-pages-articles.txt ~/wikipedia/processed/arzwiki-20150329-pages-articles.txt docs_aligned_by_lsi/

def usage():
	print 'Usage: ', sys.argv[0], '<source corpus file> <target corpus file> <output path>'
	print 'python aligning-docs-by-lsi-demo.py ~/wikipedia/processed/arwiki-20150311-pages-articles.txt ~/wikipedia/processed/arzwiki-20150329-pages-articles.txt docs_aligned_by_lsi/'
##################################################################

if len(sys.argv) < 4: usage(); sys.exit(2)


'''
This software is a demo aligning Arabic and Egyptian wikipeida comparable documents using interlanguage links. The method is described in 

https://sites.google.com/site/motazsite/Home/publications/saad_phd.pdf

Motaz Saad. Mining Documents and Sentiments in Cross-lingual Context. PhD thesis, Université de Lorraine, January 2015.

'''

import imp
tp = imp.load_source('textpro', 'textpro.py')


def main(argv):
	source_corpus_file = sys.argv[1]
	target_corpus_file = sys.argv[2]
	output_path = sys.argv[3]
	
	# step 1: train the LSI model
	source_corpus = tp.load_corpus(source_corpus_file, 'parallel')
	target_corpus = tp.load_corpus(target_corpus_file, 'parallel')
	
	merged_corpus = tp.merge_source_target_docs(source_corpus, target_corpus)
	tp.prepare_gensim_corpus('un', merged_corpus, output_path)
	
#	# step 2: 
#	build_lsi_model(corpus_name, corpus_path, topics=300)
#	
#	# step 3:
#	(corpus_path, source_corpus_name, target_corpus_name, source_language, target_language, model_path, model_name, output_path, top_n=20, doc_separator=x_seperator)


##################################################################


if __name__ == "__main__":
	main(sys.argv)



