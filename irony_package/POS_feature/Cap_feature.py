# Captalization features: The number of word with initial caps and all caps and the number of POS tags with at least initial caps.
# The result will import into MongoDB or print in the terminal and stored in Cap_result.csv
import csv
from pymongo import MongoClient
id_list = []

TagList = ['N','O','S','^','Z','L','M','V','A','R',
		   '!','D','P','&','T','X','Y','#','@','~',
		   'U','E','$',',','G']

CapDict_list = []
with open('id_text.csv', 'rb') as id_text:
	r = csv.DictReader(id_text)
	for row in r:
		id_list.append(row['tweet_id'])

with open('./ark-tweet-nlp-0.3.2/output.txt','r') as f:
	file = f.readlines()

for lines, i in zip(file[0:], range(len(id_list))):
	CapDict = {key:0 for key in TagList}
	CapDict['ini_cap_number'] = 0
	CapDict['all_cap_number'] = 0

	sentence = lines.split('\t')[0].split(' ')
	tags = lines.split('\t')[1].split(' ')


	for i in range(0,len(sentence)):
		word = sentence[i]

		if word[0].isupper():
			CapDict['ini_cap_number'] += 1
			CapDict[tags[i]] += 1

		if word.isupper():
			CapDict['all_cap_number'] += 1

	CapDict['tweet_id'] = id_list[i]
	CapDict_list.append(CapDict)

# Connect to MongoDB
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
dbtweets = db.tweets


for i in range(len(CapDict_list)):
	tweet_id = CapDict_list[i]['tweet_id']

	for a in TagList:			
		a_cap_count = a+'_cap_count'
		a_value = CapDict_list[i][a]
		if a== '$':
			a_cap_count = 'numeral_pos_cap'

		result = dbtweets.update_one({"tweet_id": tweet_id},
					{
					    "$set": {
			                a_cap_count: a_value
						}
					}
				)

	ini_cap_number = CapDict_list[i]['ini_cap_number']
	all_cap_number = CapDict_list[i]['all_cap_number']

	result = dbtweets.update_one({"tweet_id": tweet_id},
					{
					    "$set": {
			                'all_cap_number': all_cap_number,
			                'ini_cap_number': ini_cap_number
			        	}
					}
				)
"""	
with open('Cap_result.csv', 'wb') as f:
	w = csv.DictWriter(f, fieldnames=CapDict_list[0].keys())
	w.writeheader()
	w.writerows(CapDict_list)
"""