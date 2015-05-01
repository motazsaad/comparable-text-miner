# coding: utf-8

import sys
##################################################################
def usage():
	print 'Usage: ', sys.argv[0], '<inputfile> <outputfile>'
##################################################################

if len(sys.argv) < 3: usage(); sys.exit(2)


'''
Demo of Arabic morphological analysis tools, which are as follows:

- Text processing : remove Arabic diacritics and punctcutions, and tokenizing 
- Rooting (ISRI Arabic Stemmer)
- light stemming



ISRI Arabic Stemmer is described in:

Taghva, K., Elkoury, R., and Coombs, J. 2005. Arabic Stemming without a root dictionary. Information Science Research Institute. University of Nevada, Las Vegas, USA.

The difference between ISRI Arabic Stemmer and The Khoja stemmer is that ISRI stemmer does not use root dictionary. Also, if a root is not found, ISRI stemmer returned normalized form, rather than returning the original unmodified word.


Light stemming for Arabic words is to remove common affix (prefix and suffix) from words, but it does not convert words into their root form.

Example Usage:

python arabic-morphological-analysis-demo.py test-text-files/test-in.ar.txt test-text-files/test-out.ar.txt

'''

import imp
tp = imp.load_source('textpro', 'textpro.py')




def main(argv):
	inputfile = sys.argv[1]
	outputfile = sys.argv[2]
	text = open(inputfile).read().decode('utf-8')
	word_list = tp.process_text(text) # remove diacritics and punctcutions, stopwords, and tokenize text
	
	roots = tp.getRootAr(word_list) # apply rooting algorithm
	lightStems = tp.lightStemAr(word_list) # apply light stemming 
	
	output_text = 'apply rooting\n' + roots + '\n\n=================\napply light stemming\n' + lightStems
	output = open(outputfile, 'w')
	print>>output, output_text.encode('utf-8')
	output.close()


##################################################################


if __name__ == "__main__":
	main(sys.argv)



