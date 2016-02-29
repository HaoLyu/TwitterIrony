# Fix the data problems in MongoDb
def remove_sarcsam_tag():

def binarize_unigrams_bigrams():
	

if __name__ == '__main__':
	try:
		if sys.argv[1] == 'remove_sarcsam_tag':
			remove_sarcsam_tag()
		elif sys.argv[1] == 'binarize_unigrams_bigrams':
			binarize_unigrams_bigrams()

		else:
			print 'other mode'

	except IndexError:
		print 'try again and add mode arguments'