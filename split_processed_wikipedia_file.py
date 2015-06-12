# coding: utf-8


# Author: Motaz Saad

import sys, os
from os.path import basename
import logging
logging.basicConfig(format='%(levelname)s : %(asctime)s : %(message)s', level=logging.INFO)

x_seperator = '\nXXXXXXX\n' # define document separator (7 Xs). This separator is used when all the docs are in one file (a corpus file)

##################################################################


def usage():
	print 'Usage: ', sys.argv[0], '<corpus file> <number of parts> <output path>'

##################################################################

if len(sys.argv) < 3: usage(); sys.exit(2)


import imp
tp = imp.load_source('textpro', 'textpro.py')


def main(argv):
	corpus_file = sys.argv[1]
	num_parts = int(sys.argv[2])
	output_path = sys.argv[3]
	
	if not output_path.endswith('/'): output_path = output_path + '/'
	tp.check_dir(output_path) # if directory does not exist, then create
	
	docs = tp.split_wikipedia_docs_into_array(corpus_file)
	logging.info( 'corpus is loaded')
	
	parts = tp.split_list(docs, num_parts)
	
	f_name = basename(corpus_file).split()[0]
		
	for i in range(len(parts)):
		out = open(output_path + 'part-' + str(i) + '-' + f_name , 'w')
		for d in parts[i]:
			print>>out, d.encode('utf-8')
		out.close()
		logging.info('part %d is done', i )
		

##################################################################


if __name__ == "__main__":
	main(sys.argv)



