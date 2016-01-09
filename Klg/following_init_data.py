import csv	
from pymongo import MongoClient

# Store bamman.csv into dictionary
mydict = {}
mydict['tweet_id'] = []
mydict['sarcasm_score'] = []
mydict['author_full_name'] = []
mydict['author_id'] = []
mydict['tweet_text'] = []
with open('../Tweet_Data/bamman.csv','r') as f:
	next(f)
	r = csv.reader(f)
	for row in r:
		a, b, c ,d ,e = row[0].split('\t', 4)
		mydict['tweet_id'].append(a)
		mydict['sarcasm_score'].append(b)
		mydict['author_full_name'].append(c)
		mydict['author_id'].append(d)
		mydict['tweet_text'].append(e)

f.close()

# Load the data into mongodb
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
tweets = db.tweets
tweets.remove({})
for i in range(len(mydict['tweet_id'])):
	onetweet = {"tweet_id": mydict['tweet_id'][i], 
				"sarcasm_score": mydict['sarcasm_score'][i],
				"author_full_name": mydict['author_full_name'][i],
				"author_id": mydict['author_id'][i],
				"tweet_text": mydict['tweet_text'][i]
				}
	tweets.insert_one(onetweet)

