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
		# check following count and list

		one_author = tweets.find()[i]
		check1 = one_author["following_count"]
		check2 = one_author["following_list"]
		#for j in range(len(check2)):
		#	check2[j] = check2[j].encode('utf-8')

		mydict['tweet_id'].append(one_author["tweet_id"])
		mydict['sarcasm_score'].append(one_author["sarcasm_score"])
		mydict['author_full_name'].append(one_author["author_full_name"])
		mydict['author_id'].append(one_author["author_id"])
		mydict['tweet_text'].append(one_author["tweet_text"].encode('utf-8'))
		mydict['following_count'].append(one_author["following_count"])
		#mydict['following_list'].append(check2)
		mydict['following_list'].append(one_author["following_list"])
	except KeyError:
		one_author = [	tweets.find()[i]["tweet_id"], tweets.find()[i]["sarcasm_score"],
						tweets.find()[i]["author_full_name"], tweets.find()[i]["author_id"],
						tweets.find()[i]["tweet_text"]
						]

		mydict['tweet_id'].append(one_author[0])
		mydict['sarcasm_score'].append(one_author[1])
		mydict['author_full_name'].append(one_author[2])
		mydict['author_id'].append(one_author[3])
		mydict['tweet_text'].append(one_author[4].encode('utf-8'))
		mydict['following_count'].append("Not Available")
		mydict['following_list'].append("Not Available")

with open('output.csv', 'wb') as f:  
	w = csv.DictWriter(f, mydict.keys())
	w.writeheader()
	for i in range(tweets.find().count()):
		try:
			w.writerow({'tweet_id':mydict['tweet_id'][i], 
	    				'sarcasm_score':mydict['sarcasm_score'][i],
	    				'author_full_name':mydict['author_full_name'][i],
	    				'author_id':mydict['author_id'][i],
	    				'tweet_text':mydict['tweet_text'][i],
	    				'following_count':mydict['following_count'][i],
	    				'following_list':mydict['following_list'][i]
	    				})
		except UnicodeEncodeError:
			print "this No. %d row" %i
			print  mydict['following_list'][i]
			break
# Example: Read output.csv. 
# Warning: this output.csv file is 234.5MB large and require sys.maxsize
# import csv
# import getpass
# import sys
# csv.field_size_limit(sys.maxsize)
# test_file = open('output.csv', 'rb')
# csv_file = csv.DictReader(test_file, delimiter=',', quotechar='"')
# for line in csv_file:
#	print line['tweet_id'], line['following_count']
# for num, row in zip(range(2), csv_file):
# 	print line['tweet_id'], line['following_count']
