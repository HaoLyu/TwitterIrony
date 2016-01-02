# This file count the number of words with only alphabetic characters but no vowels in a tweet. 
import re

# Return the number of words with only alphabetic characters but no vowels in a tweet. 
def count_number_no_vowels(text):
	count = 0
	sentences = re.split(r'((?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s)',text)
	
	for sentence in sentences:
		for word in sentence.split(' '):
			if (len(re.findall('[a-zA-z]', word))>0) & (len(re.findall(r'([aeiouyAEIOUY]+)', word))<1):
				count += 1
	
	return count

def count_number_Polysyllables(text):
	count = 0
	syll = lambda w:len(''.join(c if c in"aeiouyAEIOUY" else' 'for c in w.rstrip('e')).split())
	sentences = re.split(r'((?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s)',text)
	
	for sentence in sentences:
		for word in sentence.split(' '):
			if (len(re.findall('[a-zA-z]', word))>0) & (syll(word)>3):
				count += 1
	
	return count




if __name__ == "__main__":
	x =	count_number_no_vowels("I like this. But I don't like that btw ntermediaries #irony.")
	print 'number of words which have no vowels is:', x
	
	x = count_number_Polysyllables("I like this. But I don't like that intermediaries btw #irony.")
	
	print 'number of polysyllables is:', x

