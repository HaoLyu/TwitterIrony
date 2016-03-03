# Add historical tweets to MongoDB
# You need to figure where your data's path is  
# Run by: Python add_hist_tweets_toMongo.py historical_data_path
import csv 
import sys
from pymongo import MongoClient

# Load the data into mongodb
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
dbtweets = db.tweets

input_file = '../../TwitterIrony_data/' + sys.argv[1]
csv.field_size_limit(sys.maxsize)

# Update hist_list of each author in MongoDB
with open(input_file,'rb') as f:
	r = csv.DictReader(f)
	keys = r.fieldnames
	for row in r:
		author_full_name = row[keys[1]]
		author_hist_list = row[keys[2]].split(", u")
		author_hist_list[0] = author_hist_list[0].strip('[]')[1:]
		author_hist_list[-1] = author_hist_list[-1].strip('[]')
		author_number = dbtweets.find({"author_full_name": author_full_name}).count()
		for i in range(author_number):
			author_tweet_id = dbtweets.find({"author_full_name": author_full_name})[i]["tweet_id"]
			result = dbtweets.update({"tweet_id": author_tweet_id},
					{
					    "$set": {
			                "hist_list": author_hist_list
			        	}
					}
				)

# Check MongoDb and transform all hist_list whose type is unicode into list
for i in range(dbtweets.find({'hist_list':{'$exists':True}}).count()):
	hist_list_type = type(dbtweets.find({'hist_list':{'$exists':True}})[i]['hist_list'])
	if hist_list_type is not type([]):
		tweet_id = dbtweets.find({'hist_list':{'$exists':True}})[i]['tweet_id']
		hist_list = dbtweets.find({'hist_list':{'$exists':True}})[i]['hist_list']
		hist_list = hist_list.encode('utf-8')
		hist_list = hist_list.split(", u")
		hist_list[0] = hist_list[0].strip('[]')[1:]
		hist_list[-1] = hist_list[-1].strip('[]')
		result = dbtweets.update({"tweet_id": tweet_id},
					{
					    "$set": {
			                "hist_list": hist_list
			        	}
					}
				)



