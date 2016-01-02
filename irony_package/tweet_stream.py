# Produce 1000 tweets with hashtag "#irony" 
# Insert each tweet into the Mongodb databse 'Tweets' after cleaning them
# I delete tweets which is RT or less than 4 words or not english or containing 'http'or 'https' or 'RT'
# Run:
# python tweet_stream.py --account 0

# import modules
import sys
import tweepy
import json
import pdb
from pymongo import MongoClient
import re
import argparse
import time

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
tweets = db.tweets

# Function 'NoWord' helps to find tweets that doesn't contain specific word
def NoWord(text, word):

	if (word in text):
		return False
	else:
		return True

# Query 1000 tweets which contain hashtag "Irony" or "Sarcasm"
#IronyTweet = tweepy.Cursor(api.search, q="#Irony").items(30000)
#IronyTweet = tweepy.Cursor(api.search, q="#Sarcasm").items(1000)


IronyTweet = tweepy.Cursor(api.search, q="#sarcasm").items(100)
while True:
	x = IronyTweet.next().id
	y = IronyTweet.next().text
	z = IronyTweet.next().in_reply_to_status_id
	print x, "this is x;", y
	print "in_reply to status id: ",z

while True:
	try:
		tweet = IronyTweet.next()
		if((tweet.retweeted == False) & 
		   (tweet.lang == 'en') & 
		   (len(tweet.text.split()) > 4) & 
		   (NoWord(tweet.text, "http://") == True) &
		   (NoWord(tweet.text, "https://") == True) &
		   (NoWord(tweet.text, "RT") == True) &
		   (tweets.find({"author": str(tweet.author.screen_name)}).count() < 1)): 		 
		
			onetweet = {"author": str(tweet.author.screen_name), 
						"text": tweet.text
						}	

			print onetweet
			tweets.insert_one(onetweet)				
		else:
			print tweet.text

	except tweepy.error.TweepError:
		print "Twitter rate limit, need to wait 15 min"
		time.sleep(60 * 16)
		continue
	except StopIteration:
		break



print "total count in tweets collection is: %d " % tweets.find({}).count()


