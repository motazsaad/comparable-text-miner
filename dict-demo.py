# coding: utf-8

import sys
##################################################################
def usage():
	print 'Usage: ', sys.argv[0], '<inputfile> <outputfile> <source language>'
##################################################################

if len(sys.argv) < 4: usage(); sys.exit(2)


'''
Demo of Arabic-English dictrationry translation using Open Multilingual WordNet (OMW)

Dictionaries are obtained from Open Multilingual WordNet website: http://compling.hss.ntu.edu.sg/omw/

# To cite these dictionaries:
# Francis Bond and Kyonghee Paik (2012), A survey of wordnets and their licenses In Proceedings of the 6th Global WordNet Conference (GWC 2012). Matsue. 64–71.
# Francis Bond and Ryan Foster (2013), Linking and extending an open multilingual wordnet. In 51st Annual Meeting of the Association for Computational Linguistics: ACL-2013. Sofia. 1352–1362. 

'''

import imp
tp = imp.load_source('textpro', 'textpro.py')




def main(argv):
	inputfile = sys.argv[1]
	outputfile = sys.argv[2]
	source_language = sys.argv[3]
	text = open(inputfile).read().decode('utf-8')
	print text
	text = tp.process_text(text)
	result = [] 
	for word in text:
		translations = []
		if source_language == 'ar': translations = tp.translate_ar2en(word)
		if source_language == 'en': translations = tp.translate_en2ar(word)
		for t in translations: result.append(t.strip())

	output_text = ' '.join(result)
	output = open(outputfile, 'w')
	print>>output, output_text
	output.close()


##################################################################


if __name__ == "__main__":
	main(sys.argv)



