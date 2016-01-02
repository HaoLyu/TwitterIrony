import sys
import json
from pymongo import MongoClient
import AllTweets as AllT
import Tweet_Transfer_BOW as BOW

# Connect to MongoDB
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
dbtweets = db.tweets


def original_message_unigram_fature(tweet_id):

	author_tweet = dbtweets.find_one({'tweet_id': tweet_id})
	original_text = author_tweet['in_reply_to_status_id_text']

	if(original_text == None):
		#print 'this tweet was not replying to other tweet'
		return 
	else:
		original_text_unigram = BOW.Get_unigrams(AllT.collect_original_messages(), original_text)
		return original_text_unigram

if __name__ == "__main__":

	foo = original_message_unigram_fature()
	print foo