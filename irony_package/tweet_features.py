# This file is used to generate tweet features of one author
# Features included: intensifier, unigram, bigram, pronunciation features

import sys
sys.path.append('Pronunciation_feature')
import Pronunciation_feature as Pron
import tweepy
import json
import argparse
import operator
import datetime
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import LDA_topic as LDA
import Tweet_Transfer_BOW as BOW
import AllTweets as AllT
from pymongo import MongoClient
from scipy import sparse
import re
import pickle


# Connect to MongoDB
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
dbtweets = db.tweets

"""

# Class User contains all the profile features of the user
class User(object):

	def __init__(self, userid):
		self.userid = userid
		self.user = api.get_user(self.userid)
		self.user_timeline = tweepy.Cursor(api.user_timeline, id=self.userid)


# Get top 100 terms with highest tf-idf scores
	def get_top_100_tfidf_terms(self):
		tweetList = []
		top_tfidf_list = []

		for tweet in self.get_user_tweets():
			strTweet = str((tweet.text).encode('ascii', 'ignore'))
			tweetList.append(strTweet)
			
		vectorizer = TfidfVectorizer(analyzer='word', min_df=0, ngram_range=(1,3), stop_words='english')
		tfidf_matrix = vectorizer.fit_transform(tweetList)
		idf = vectorizer.idf_
		scores =  dict(zip(vectorizer.get_feature_names(), idf))
		sortedList = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
		
		for n in range(0,100):
			top_tfidf_list.append(sortedList[n][0])
		
		return top_tfidf_list

# Get the proportions of top 100 topics 
	def get_topics_proportions(self):

		UserTopicPor = LDA.HistoryTopic(self.userid) 
		return(UserTopicPro.get_topics_proportions())


"""
# Get the binary indicatior for whether the tweet contains a word in intensifiers
def get_intensifiers(text):
	inten_file = open('intensifiers.txt', 'r')
	intensifier = 0

	for line in inten_file.readlines():
		regex = r"\b" + re.escape(line) + r"\b"

		if re.findall(regex, text, re.IGNORECASE):
			intensifier = 1
			break
		else:
			continue
	
	return intensifier



#--------
# unigrams
# all_unigrams is the bag of word in unigrams
all_unigrams = BOW.Get_unigrams_bigrams(AllT.collect_text())[0]
vect1 = CountVectorizer(vocabulary=all_unigrams)
# bigrams
# all_bigrams is the bag of word in bigrams
all_bigrams = BOW.Get_unigrams_bigrams(AllT.collect_text())[1]  
vect2 = CountVectorizer(vocabulary=all_bigrams)
start_time = datetime.datetime.now()


for i in range(dbtweets.find().count()):
	cur_time = datetime.datetime.now()
	delta = cur_time - start_time
	#print 'this is the ', i+1, 'tweet', 'total time is: ', delta
	tweet_id = dbtweets.find()[i]['tweet_id']
	tweet_text = dbtweets.find()[i]['tweet_text']

	
	number_no_vowels = Pron.count_number_no_vowels(tweet_text)
	number_Polysyllables = Pron.count_number_Polysyllables(tweet_text)
	unigrams = vect1.transform([tweet_text]).toarray()
	for uu in xrange(unigrams[0].shape[0]):
		if unigrams[0][uu]>1:
			unigrams[0][uu] = 1
	S_uni = sparse.csr_matrix(unigrams)
	serialized_uni = pickle.dumps(S_uni, protocol=0)
	bigrams = vect2.transform([tweet_text]).toarray()
	for bb in xrange(bigrams[0].shape[0]):
		if bigrams[0][bb]>1:
			bigrams[0][bb] = 1
	S_bi = sparse.csr_matrix(bigrams)
	serialized_bi = pickle.dumps(S_bi, protocol=0)
	intensifier = get_intensifiers(tweet_text)

	result = dbtweets.update_one({"tweet_id": tweet_id},
			{
			    "$set": {
	                "intensifier": intensifier,
	                "word_unigrams": serialized_uni,
	                "word_bigrams": serialized_bi,
					 "number_Polysyllables ": number_Polysyllables,
					 "number_no_vowels": number_no_vowels
	        	}
			}
		)
		
	


