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
		tweet_text = tweet_text.replace('#Sarcasm','').strip()
		tweet_text = tweet_text.replace('#sarcastic','').strip()
		tweet_text = tweet_text.replace('#Sarcastic','').strip()
		result = dbtweets.update_one({"tweet_id": tweet_id},
				{
				    "$set": {
		                "tweet_text": tweet_text
		        	}
				}
			)
def check_intensifier_tag():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets
	d = {}
	
	
	for i in range(dbtweets.find({'intensifier':{'$exists':True}}).count()):
		intensifier = dbtweets.find({'intensifier':{'$exists':True}})[i]['intensifier']
		
		if intensifier in d:
			continue
		else:
			d[intensifier] = 1

	print d
		
	

if __name__ == '__main__':
	try:
		if sys.argv[1] == 'remove_sarcasm_tag':
			remove_sarcasm_tag()
		elif sys.argv[1] == 'check':
			check_intensifier_tag()

		else:
			print 'other mode'

	except IndexError:
		print 'try again and add mode arguments'