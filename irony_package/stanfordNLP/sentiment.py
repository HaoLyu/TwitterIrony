# convert data from collection 'test' to 'tweets'
import csv
from pymongo import MongoClient
# Load the data into mongodb
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
dbtest = db.test
mydict = {}
mydict['tweet_id'] = []
mydict['tweet_whole_sentimentpredict'] = []
mydict['positivenode'] = []
mydict['negativenode'] = []
mydict['sentiment_distance'] = []
mydict['max_word_senti'] = []
mydict['min_word_senti'] = []
mydict['effect_distance'] = []
mydict['max_word_effect'] = []
mydict['min_word_effect'] = []


count_number = dbtest.find().count()
for i in range(count_number):
	query = dbtest.find()[i]
	mydict['tweet_id'].append(query['tweet_id'])
	mydict['tweet_whole_sentimentpredict'].append(query['tweet_whole_sentimentpredict'])
	mydict['positivenode'].append(query['positivenode'])
	mydict['negativenode'].append(query['negativenode'])
	mydict['sentiment_distance'].append(query['sentiment_distance'])
	mydict['max_word_senti'].append(query['max_word_senti'])
	mydict['min_word_senti'].append(query['min_word_senti'])
	mydict['effect_distance'].append(query['effect_distance'])
	mydict['max_word_effect'].append(query['max_word_effect'])
	mydict['min_word_effect'].append(query['min_word_effect'])

dbtweets = db.tweets
for i in range(len(mydict['tweet_id'])):
	result = dbtweets.update_one({"tweet_id": mydict['tweet_id'][i] },
			{
			    "$set": {
	                "tweet_whole_sentimentpredict": mydict['tweet_whole_sentimentpredict'][i],
	                "positivenode": mydict['positivenode'][i],
					"negativenode": mydict['negativenode'][i],
					"sentiment_distance": mydict['sentiment_distance'][i],
					"max_word_senti": mydict['max_word_senti'][i],
					"min_word_senti": mydict['min_word_senti'][i],
					"effect_distance": mydict['effect_distance'][i],
					"max_word_effect": mydict['max_word_effect'][i],
					"min_word_effect": mydict['min_word_effect'][i]


	        	}
			}
		)