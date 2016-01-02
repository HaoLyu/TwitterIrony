# This file is used to generate all features of one author
# Run:
# python tweet_features.py --account 0

# import modules
import sys
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

# Class User contains all the profile features of the user
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


# Class IronyTweet contains all the features of the inronic tweet
class IronyTweet(object):

	def __init__(self, userid, tweet):
		self.userid = userid
		self.tweet = tweet

# Get the count of word unigrams and bigrams
	def get_unigrams_bigrams_count(self):
		all_unigrams = BOW.Get_unigrams_bigrams(AllT.collect_text())[0]  
		all_bigrams = BOW.Get_unigrams_bigrams(AllT.collect_text())[1]  

		vect1 = CountVectorizer(vocabulary=all_unigrams)
		unigrams = vect1.fit_transform(self.tweet).toarray()

		vect2 = CountVectorizer(vocabulary=all_bigrams)
		bigrams = vect2.fit_transform(self.tweet).toarray()

		return (unigrams, bigrams)

# Get the binary indicatior for whether the tweet contains a word in intensifiers
	def get_intensifiers(self):
		inten_file = open('intensifiers.txt', 'r')
		intensifier = 0

		for line in inten_file.readlines():
			regex = r"\b" + re.escape(line) + r"\b"

			if re.findall(regex, self.tweet, re.IGNORECASE):
				intensifier = 1
				break
			else:
				continue
		
		return intensifier
	
# print test
#newtweet = IronyTweet('_turgon', "In this country, \"democracy\" means pro-government. #irony")

#print newtweet.get_unigrams_bigrams_count()[0]

newuser = User('spykeezy')
print newuser.get_profile
#print newuser.get_top_100_tfidf_terms()
"""
print api.search_users('spykeezy')[0].location
print api.search_users('spykeezy')[0].favourites_count
print api.search_users('spykeezy')[0].followers_count
print api.search_users('spykeezy')[0].time_zone
print api.search_users('spykeezy')[0].friends_count
print api.search_users('spykeezy')[0].created_at
print api.search_users('spykeezy')[0].statuses_count
print api.search_users('spykeezy')[0].screen_name
print (api.search_users('spykeezy')[0].name).encode('utf8')
print api.search_users('spykeezy')[0].name
print '-'*40

"""

