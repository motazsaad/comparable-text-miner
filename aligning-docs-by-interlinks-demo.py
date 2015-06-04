# coding: utf-8


# Author: Motaz Saad

import sys
##################################################################
#Example Usage:
#python aligning-docs-by-interlinks-demo.py ~/wikipedia/processed/arwiki-20150311-pages-articles.txt ~/wikipedia/processed/arzwiki-20150329-pages-articles.txt ar arz docs_aligned_by_links/

def usage():
	print 'Usage: ', sys.argv[0], '<source corpus file> <target corpus file> <source language> <target language> <output path>'
	print 'python aligning-docs-by-interlinks-demo.py ~/wikipedia/processed/arwiki-20150311-pages-articles.txt ~/wikipedia/processed/arzwiki-20150329-pages-articles.txt ar arz docs_aligned_by_links/'
##################################################################

if len(sys.argv) < 6: usage(); sys.exit(2)


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
	source_language = sys.argv[3]
	target_language = sys.argv[4]
	output_path = sys.argv[5]
	
	tp.aligning_documents_by_interlanguage_links(source_corpus_file, target_corpus_file, source_language, target_language, output_path)


##################################################################


if __name__ == "__main__":
	main(sys.argv)



