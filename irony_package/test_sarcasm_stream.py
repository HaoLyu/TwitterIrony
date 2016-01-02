# Crawl all tweets in test_sarcasm.tsv using the status id by Tweepy
# Insert all tweets into MongoDB, including the status_id, text, user name, judge of each tweet.
# Run:
# python tweet_stream.py --account 0

# import modules
import sys
import tweepy
import json
import pdb
from pymongo import MongoClient
import re
from sklearn.feature_extraction.text import CountVectorizer
import argparse
import csv
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

# Connect to MongoDB
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
testSarcasm = db.testSarcasm

# Authentication, API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

# Open test_sarcasm file and sotre test tweets in MongoDB
with open('test_sarcasm.tsv','rb') as tsv:
	for line in csv.reader(tsv, dialect='excel-tab'):
		testSarcasm_status_id =  line[0]
		testSarcasm_judge = line[1]

		try:
			if(testSarcasm.find({"status_id": testSarcasm_status_id}).count() < 1):

				tweet = api.get_status(id=testSarcasm_status_id)
				testSarcasm_text = tweet.text
				testSarcasm_user = tweet.user.screen_name

				onetweet = {"status_id": testSarcasm_status_id,
							"text": testSarcasm_text,
							"user_name": testSarcasm_user, 
							"judge": testSarcasm_judge
							} 

				print onetweet
				testSarcasm.insert_one(onetweet) 

		except tweepy.error.TweepError as e:
			
			if(e.message[0]['code'] != 88):
				print "No status found with this id %s" % (testSarcasm_status_id)
				print e
				print e.message[0]['code']
				continue

			else:
				print "Twiter rate limit, need to wait 15 min"
				time.sleep(60*16)
				continue

		except StopIteration:
			break
		



