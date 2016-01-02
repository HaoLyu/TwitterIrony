import sys
import tweepy
import json
from pymongo import MongoClient
import argparse
import time
import LDA_topic as LDA
import numpy as np
import lda
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
dbtweets = db.tweets

# get the historical tweets of user with id from Twitter
def get_context(id):
	context_tweets = []
	for status in tweepy.Cursor(api.user_timeline, id = id).items(300):
		context_tweets.append(status.text)

	return context_tweets

def interaction_topics_fature(tweet_id):

	author_tweet = dbtweets.find_one({'tweet_id': tweet_id})
	author_name = author_tweet['author_name']
	addressee_id = author_tweet['in_reply_to_user_id']


	if(addressee_id == None):
		print 'this tweet was not replying to other users'
		return 
	else:
		author_historical_tweets = author_tweet['context']
		addressee_historical_tweets = get_context(addressee_id)
		LDATopic = LDA.HistoryTopic(author_name)
		interaction_topics = LDATopic.get_interaction_topics_proportions(author_historical_tweets, addressee_historical_tweets)
		
		return interaction_topics


if __name__ == "__main__":

	foo = get_context('BarackObama')
	print "the numebr of all context tweets is: ", len(foo)
	for line in foo:
		print line
		print '\n'