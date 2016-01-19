# Convert data from MongoDB to csv
from pymongo import MongoClient
import csv
# Load the data into mongodb
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
tweets = db.tweets
mydict = {}
mydict['tweet_id'] = []
mydict['sarcasm_score'] = []
mydict['author_full_name'] = []
mydict['author_id'] = []
mydict['tweet_text'] = []
mydict['following_count'] = []
mydict['following_list'] = []


for i in range(tweets.find().count()):
	try:
		one_author = [	tweets.find()[i]["tweet_id"], tweets.find()[i]["sarcasm_score"],
						tweets.find()[i]["author_full_name"], tweets.find()[i]["author_id"],
						tweets.find()[i]["tweet_text"], tweets.find()[i]["following_count"],
						tweets.find()[i]["following_list"]
						]

		mydict['tweet_id'].append(one_author[0])
		mydict['sarcasm_score'].append(one_author[1])
		mydict['author_full_name'].append(one_author[2])
		mydict['author_id'].append(one_author[3])
		mydict['tweet_text'].append(one_author[4])
		mydict['following_count'].append(one_author[5])
		mydict['following_list'].append(one_author[6])
	except KeyError:
		one_author = [	tweets.find()[i]["tweet_id"], tweets.find()[i]["sarcasm_score"],
						tweets.find()[i]["author_full_name"], tweets.find()[i]["author_id"],
						tweets.find()[i]["tweet_text"]
						]

		mydict['tweet_id'].append(one_author[0])
		mydict['sarcasm_score'].append(one_author[1])
		mydict['author_full_name'].append(one_author[2])
		mydict['author_id'].append(one_author[3])
		mydict['tweet_text'].append(one_author[4])
		mydict['following_count'].append("Not Available")
		mydict['following_list'].append("Not Available")

with open('output.csv', 'wb') as f:  # Just use 'w' mode in 3.x
    w = csv.DictWriter(f, mydict.keys())
    w.writeheader()
    w.writerow(mydict)