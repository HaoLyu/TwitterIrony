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
from sklearn.feature_extraction.text import CountVectorizer
import argparse
import csv
import datetime
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
"""
# Open test_sarcasm file and sotre test tweets in MongoDB
with open('test_sarcasm.tsv','rb') as tsv:
	for line in csv.reader(tsv, dialect='excel-tab'):
		testSarcasm_status_id =  line[0]
		testSarcasm_judge = line[1]
		
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

"""

"""
vect = CountVectorizer(ngram_range=(1,1), stop_words='english', token_pattern=r'\b\w+\b',min_df=1)
analyze = vect.build_analyzer()
x = analyze('This is an example tweet #irony')
y = 'such a good tweet new,irony but it is not irony'
vect1 = CountVectorizer(vocabulary=x)
newfit = vect1.fit_transform([y])
print x
print newfit.toarray()
"""
class User(object):

	def __init__(self, userid):
		self.userid = userid
		self.user = api.get_user(self.userid)
		self.user_timeline = tweepy.Cursor(api.user_timeline, id=self.userid)

# Get the followers' screen name
	def get_user_follower(self):
		user_follower = []

		for friend in self.user.followers():
			user_follower.append(friend.screen_name)
		
		return user_follower

# Get the followings' screen name
	def get_user_following(self):
		user_following = []

		for friend in self.user.friends():
			user_following.append(friend.screen_name)
		
		return user_following
	
# Get all the tweets of this user
	def get_user_tweets(self):
		user_tweets = []

		for item in self.user_timeline.items(2000):
			user_tweets.append(item)
		
		return user_tweets

# test function
	def test(self):
		return self.user.name

# Get the gender of the user
	def get_user_gender(self):
		return self

# Get the profile description of the user
	def get_profile(self):
		return self.user.description

# Get the number of status
	def get_num_of_status(self):
		self.statuses_count = self.user.statuses_count
		return self.statuses_count 

# Get the user's duration on Tiwtter 
	def duration(self):
		start_time = self.user.created_at
		current_time = datetime.datetime.now()
		duration = (current_time - start_time).days
		self.duration = duration

		return self.duration

# Get the average number of posts per day
	def ave_num_post(self):
		ave_num_post = round(float(self.statuses_count)/float(self.duration), 5)

		return ave_num_post

# Get the time zone of the user
	def get_time_zone(self):
		return self.user.time_zone

# Get the status of user's verification
	def get_verified(self):

		if self.user.verified:
			return 'verified'

		else:
			return 'unvertified'

"""
newuser = User('_turgon')
print newuser.test()
print newuser.get_profile()
"""
inten_list = []
inten_file = open('intensifiers.txt','r')
for line in inten_file.readlines():
	print type(line)

print len(inten_list)
