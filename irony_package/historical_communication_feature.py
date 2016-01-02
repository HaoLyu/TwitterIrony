import sys
import tweepy
import json
from pymongo import MongoClient
import argparse
import numpy as np
import Author_Addressee_interactional_topics as AA
import collections
from operator import itemgetter

# Connect to MongoDB
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
dbtweets = db.tweets


# Count the number of previous messages sent from the author to the addressee
def number_of_messages(tweet_id):
	number_of_messages = 0
	author_tweet = dbtweets.find_one({'tweet_id': tweet_id})
	author_historical_tweets = author_tweet['context']
	addressee_name = author_tweet['in_reply_to_user_name']

	for one_tweet in author_historical_tweets:
		if ("@"+addressee_name) in one_tweet:
			number_of_messages += 1
		else:
			continue

	return number_of_messages

# the rank of the addressee among the user’s @ mention recipients 
def rank_of_addressee(tweet_id):
	name_list = []
	author_tweet = dbtweets.find_one({'tweet_id': tweet_id})
	author_historical_tweets = author_tweet['context']
	addressee_name = author_tweet['in_reply_to_user_name']
	
	for one_tweet in author_historical_tweets:
		for one_token in one_tweet.split(" "):
			if "@" in one_token:
				name_list.append(one_token)
			else:
				continue

	check = ('@'+addressee_name) in name_list
	if check:
		rank_dict = collections.Counter(name_list)
		name_frequency = rank_dict.values()
		name_frequency.sort()
		# rank score is between 0-1, 0 means unfamilar, 1 means the most active
		addressee_rank = (name_frequency.index(rank_dict['@'+addressee_name]) + 1)/ float(len(rank_dict))

		return addressee_rank

	else:

		return 0

# whether or not there have been at least one (and two) mutual @-messages exchanged between the author and the addressee
def mutual_mention(tweet_id):
	author_tweet = dbtweets.find_one({'tweet_id': tweet_id})
	author_name = author_tweet['author']
	addressee_id = author_tweet['in_reply_to_user_id']

	if not rank_of_addressee(tweet_id).check:
		return 0
	else:
		addressee_context = AA.get_context(addressee_id)
		for one_tweet in addressee_context:
			for one_token in one_tweet.split(" "):
				if ("@"+author_name) in one_token:
					return 1
				else:
					continue

	return 0 




if __name__ == "__main__":
	print "run functions to get historical communication features"
