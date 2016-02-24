# Store your tweets into ark-tweet-nlp-0.3.2/examples/example_tweets.txt
# Head to directory ark-tweet-nlp-0.3.2 run  	 then we get the result file output.txt
# Second run this file: python POS_count.py
# The result will store in MongoDB or print in the terminal and stored in POS_result.csv
import csv
from pymongo import MongoClient

TagList = ['N','O','S','^','Z','L','M','V','A','R',
		   '!','D','P','&','T','X','Y','#','@','~',
		   'U','E','$',',','G']
TagDict_list = []
id_list = []

with open('id_text.csv', 'rb') as id_text:
	r = csv.DictReader(id_text)
	for row in r:
		id_list.append(row['tweet_id'])

with open('./ark-tweet-nlp-0.3.2/output.txt','r') as f:
	file = f.readlines()

for lines, i in zip(file[0:], range(len(id_list))):
	TagDict = {key:0 for key in TagList}

	tags = lines.split('\t')[1].split(' ')
	
	# Set the count of each tag
	for tag in tags:
		TagDict[tag] += 1
	
	Tag_total_count = float(sum(TagDict.values()))
	
	# Set the ratio of each tag 
	for tag in TagList:
		TagDict[tag+'_ratio'] = TagDict[tag]/Tag_total_count

	number_of_lexical_words = TagDict['N'] + TagDict['V'] + TagDict['A'] + TagDict['R']
	TagDict['lexical_density'] = 100 * number_of_lexical_words/Tag_total_count
	TagDict['tweet_id'] = id_list[i]

	TagDict_list.append(TagDict)

# Connect to MongoDB
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
dbtweets = db.tweets


for i in range(len(TagDict_list)):
	tweet_id = TagDict_list[i]['tweet_id']

	for a in TagList:			
		a_ratio = a+'_ratio'
		a_value = TagDict_list[i][a]
		a_ratio_value = TagDict_list[i][a_ratio]
		if a== '$':
			a = 'numeral_pos'
			a_ratio = a+'_ratio'

		result = dbtweets.update_one({"tweet_id": tweet_id},
					{
					    "$set": {
			                a: a_value,
			                a_ratio: a_ratio_value
			        	}
					}
				)

	lexical_density = TagDict_list[i]['lexical_density']
	result = dbtweets.update_one({"tweet_id": tweet_id},
					{
					    "$set": {
			                'lexical_density':lexical_density
			        	}
					}
				)

#with open('POS_result.csv', 'wb') as f:
#	w = csv.DictWriter(f, fieldnames=TagDict_list[0].keys())
#	w.writeheader()
#	w.writerows(TagDict_list)

# Display all tags
#for one in TagDict_list:
#	print 'this is the tag count:',one
#	print '\n'
