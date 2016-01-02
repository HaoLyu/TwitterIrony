# Function Get_BOW is used to produce Bag of Words and tokenizer counts
# from tweets documents
# Function Get_unigrams_bigrams can get the unigram word list and bigram word list
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import AllTweets as AllT


def Get_BOW(tweets):
	# import countvectorzier to do process and 
	# a built-in stop word list for English is used
	count_vect = CountVectorizer(stop_words='english')
	train_counts = count_vect.fit_transform(tweets)
	
	vocab = count_vect.vocabulary_.keys()
	train_counts = (train_counts).toarray()

	return (vocab, train_counts)

def Get_unigrams_bigrams(corpus):
	# import countvectorzier to generate unigrams and bigrams
	unicount_vect = CountVectorizer(ngram_range=(1,1), lowercase = False,  stop_words='english',  token_pattern=r'\b\w+\b', min_df=1)
	unicount = unicount_vect.fit_transform(corpus).toarray() 
	unigrams = unicount_vect.get_feature_names()


	bicount_vect = CountVectorizer(ngram_range=(2,2), lowercase = False, stop_words='english',  token_pattern=r'\b\w+\b', min_df=1)
	bicount = bicount_vect.fit_transform(corpus).toarray() 
	bigrams = bicount_vect.get_feature_names()
	
	return (unigrams, bigrams)

def Get_unigrams(corpus, tweet):
	# import countvectorzier to generate unigrams 
	unicount_vect = CountVectorizer(ngram_range=(1,1), lowercase = False,  stop_words='english',  token_pattern=r'\b\w+\b', min_df=1)
	X = unicount_vect.fit_transform(corpus)
	unigrams = unicount_vect.transform(tweet).toarray()
	unigram_names = unicount_vect.get_feature_names()
	return (unigrams, unigram_names)


all_tweets_grams = Get_unigrams_bigrams(AllT.collect_text())

if __name__ == '__main__':
	print "Running as a file, not as imported"
	print all_tweets_grams[0][0:15]
	print all_tweets_grams[1][0:15]
