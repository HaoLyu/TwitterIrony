# This file has three functions
# collect_tweets get the list 'tweets', which contain all tweetAuthors and tweetTexts
# collect_text get the list 'tweets_text', which contain all tweetText
# collect_profiles gets the list 'profiles', which contain all the profiles and tweetAuthors,
from pymongo import MongoClient
import re
# Connect to MongoDB
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
dbtweets = db.test

# Collect all tweets from MongoDB
def collect_tweets():
	tweets = []

	for n in range(dbtweets.find().count()):
		tweetAuthor = dbtweets.find()[n]['author']
		tweetText = dbtweets.find()[n]['text']
		tweets.append([tweetAuthor, tweetText])

	return tweets

def collect_topic_context():
	context = []

	for n in range(dbtweets.find().count()):
		tweet_context = dbtweets.find()[n]['context']
		context.append(tweet_context)

	return context

# Collect all tweets text 
def collect_text():
	tweets_text = []

	for n in range(dbtweets.find().count()):
		tweets_text.append(dbtweets.find()[n]['text'])

	return tweets_text

# Collect all profiles
def collect_profiles():
	profiles = []

	for n in range(dbtweets.find().count()):
		profile = dbtweets.find()[n]['profile']
		profiles.append(str(profile.encode('utf-8')))
	
	return profiles
# Collect all original messages
def collect_original_messages():
	all_original_messages =[]
	for n in range(dbtweets.find().count()):
		original_message = dbtweets.find()[n]['in_reply_to_status_id_text']
		all_original_messages.append(str(original_message.encode('utf-8')))

	return all_original_messages

if __name__ == '__main__':
	print "This is used as python file, not imported"
