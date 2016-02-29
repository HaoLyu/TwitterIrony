# Fix the data problems in MongoDb
import sys
from pymongo import MongoClient
import Tweet_Transfer_BOW as BOW
import AllTweets as AllT

def remove_sarcasm_tag():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	for i in range(dbtweets.find().count()):
		tweet_id = dbtweets.find()[i]['tweet_id']
		tweet_text = dbtweets.find()[i]['tweet_text']
		tweet_text = tweet_text.replace('#sarcasm','').strip()
		tweet_text = tweet_text.replace('#sarcastic','').strip()
		result = dbtweets.update_one({"tweet_id": tweet_id},
				{
				    "$set": {
		                "tweet_text": tweet_text
		        	}
				}
			)
def check_sarcasm_tag():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	all_unigrams = BOW.Get_unigrams_bigrams(AllT.collect_text())[0]
	print type(all_unigrams)
	
	print all_unigrams
	print '#sarcasm' in all_unigrams
	print '#sarcastic' in all_unigrams
	print 'sarcasm' in all_unigrams
	print 'sarcastic' in all_unigrams
	if 'sarcasm' in all_unigrams:
		print all_unigrams.index('sarcasm')
	if 'sarcastic' in all_unigrams:
		print all_unigrams.index('sarcastic')
	vect1 = CountVectorizer(vocabulary=all_unigrams)
	
	"""
	for i in range(dbtweets.find().count()):
		tweet_id = dbtweets.find()[i]['tweet_id']
		tweet_text = dbtweets.find()[i]['tweet_text']
		if i%200 == 0:
			print i
		c = 'sarcasm'
		d= 'sarcastic'
		a = '#sarcasm'
		b = '#sarcastic'
		if (a in tweet_text) or (b in tweet_text) or(c in tweet_text) or(d in tweet_text):
			print tweet_id, tweet_text
		
	"""

if __name__ == '__main__':
	try:
		if sys.argv[1] == 'remove_sarcasm_tag':
			remove_sarcasm_tag()
		elif sys.argv[1] == 'check':
			check_sarcasm_tag()

		else:
			print 'other mode'

	except IndexError:
		print 'try again and add mode arguments'