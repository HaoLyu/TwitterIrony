# Extract author features from Twitter and load them into MongoDB collection 'test'
# Vectorize all profiles and get the unigram matrix
# Serialize all matrix and load them into MongoDB
# Run:
# python author_feature_stream.py --account 0

# Import modules
import sys
import tweepy
import json
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
import argparse
import time
import numpy as np
import AllTweets as AT
import re
import scipy
from bson.binary import Binary
import pickle
import nltk

# Parse arguments
parser = argparse.ArgumentParser(description='Tweepy Stream.')
parser.add_argument('--account')
opts = parser.parse_args()

use_account = str(opts.account)
# Load account info from file

account_file = open('../Credentials/twitter_accounts.json', 'r')
all_accounts = json.load(account_file)
account = all_accounts[use_account]
account_file.close()


consumer_key = account['consumer_key']
consumer_secret = account['consumer_secret']
access_key = account['access_key']
access_secret = account['access_secret']

# Authentication, API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

# Connect to MongoDB
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
test = db.test

# Extract all authors' profile and load them into the database
def load_all_profile():
	n = 0 

	while n < (test.find().count()):
		try:	
			tweetAuthor = test.find()[n]["author"]
			user = api.get_user(tweetAuthor)
			author_profile = str(user.description.encode('utf-8'))
			author_profile = re.sub(r'[^\w\s]','',author_profile)
			print "this is ", n, " user: "
			n = n+1
			if(author_profile and not author_profile.isspace()):
				result = test.update_one({"author": tweetAuthor}, {"$set": {"profile": author_profile}})   
			else:
				print "None profile"
				result = test.update_one({"author": tweetAuthor}, {"$set": {"profile": " "}})   
				continue

		except tweepy.error.TweepError as e:
			print e.message[0]

			if(e.message[0]['code'] == 88):
				print "Twitter rate limit, need to wait 15 min"
				time.sleep(60 * 16)
				n = n-1
				continue

			result = test.update_one({"author": tweetAuthor}, {"$set": {"profile": " "}}) 
			n = n +1
			continue

		except StopIteration:
			break

# Extract uni-gram of each profile and update each author's profile_unigram
def update_profile_unigram():
	all_profiles = AT.collect_profiles()

	# Import countvectorzier to generate unigrams 
	unicount_vect = CountVectorizer(ngram_range=(1,1), lowercase = False,  stop_words='english',  token_pattern=r'\b\w+\b', min_df=1)
	unicount = unicount_vect.fit_transform(all_profiles).toarray() 
	unigrams = unicount_vect.get_feature_names()
	print unicount_vect
	x = nltk.cluster.api.ClusterI()
	y = x.cluster(unicount, assign_clusters=False)
	# Load profile_unigram into MongoDB
	"""
	for n in range(test.find().count()):
		tweetAuthor = test.find()[n]["author"]
		profile_unigram = scipy.sparse.coo_matrix(unicount_vect.transform([test.find()[n]["profile"]]).toarray())
		print type(profile_unigram)
		print "-"*20
		print profile_unigram
		print "-"*20
		pickle_profile_unigram = Binary(pickle.dumps(profile_unigram, protocol=2), subtype=128 )
		result = test.update_one({"author": tweetAuthor}, {"$set": {"profile_unigram": pickle_profile_unigram}})   
	"""
	"""
	# get profile_unigram from mongodb, convert Binary to numpy array
	x = pickle.loads(test.find()[22]["profile_unigram"])
	print x
	"""

if __name__ == '__main__':
	print "Running as a file, not as imported"
	#load_all_profile()
	update_profile_unigram()

