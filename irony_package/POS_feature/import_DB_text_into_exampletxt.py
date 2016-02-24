# Import tweet_id and tweet_text from MongoDB into into ark-tweet-nlp-0.3.2/examples/example_tweets.txt
from pymongo import MongoClient
import csv
import sys

# Connect to MongoDB
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
dbtweets = db.tweets

data = {key:[] for key in ['tweet_id','tweet_text']}
for i in range(dbtweets.find().count()):
	data['tweet_id'].append(dbtweets.find()[i]['tweet_id'])
	data['tweet_text'].append(dbtweets.find()[i]['tweet_text'])

with open('id_text.csv','wb') as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames=data.keys())
	writer.writeheader()
	for i in range(len(data['tweet_id'])):
		writer.writerow({'tweet_id': data['tweet_id'][i], 'tweet_text': data['tweet_text'][i].encode('utf-8')})

with open('ark-tweet-nlp-0.3.2/examples/example_tweets.txt','wb') as txtfile:
	for i in range(len(data['tweet_id'])):
		txtfile.write(data['tweet_text'][i].encode('utf-8')+'\n')




