# Update the 'hist_list' field of each author when this author has several tweets 
# and only one tweet has 'hist_list' field.
from pymongo import MongoClient

# Load the data into mongodb
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
dbtweets = db.tweets

recorded_dict = {}

for i in range(dbtweets.find({'hist_list' : {'$exists': True}}).count()):
	author_name = dbtweets.find({'hist_list' : {'$exists': True}})[i]['author_full_name']
	if author_name in recorded_dict:
		continue
	hist_list = dbtweets.find({'hist_list' : {'$exists': True}})[i]['hist_list']

	if (dbtweets.find({'author_full_name' : author_name}).count()) == 1:
		continue

	for j in range(dbtweets.find({'author_full_name' : author_name}).count()):
		tweet_id = dbtweets.find({'author_full_name' : author_name})[j]['tweet_id']
		result = dbtweets.update_one({"tweet_id": tweet_id},
						{
						    "$set": {
				                "hist_list": hist_list
				        	}
						}
					)
	recorded_dict[author_name] = 1
