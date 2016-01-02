# Store your tweets into ark-tweet-nlp-0.3.2/examples/example_tweets.txt
# Head to directory ark-tweet-nlp-0.3.2 run ./runTagger.sh example/example_tweets.txt then we get the result file output.txt
# Second run this file: python POS_count.py
# The result will print in the terminal and stored in POS_result.csv
import csv

TagList = ['N','O','S','^','Z','L','M','V','A','R',
		   '!','D','P','&','T','X','Y','#','@','~',
		   'U','E','$',',','G']
TagDict_list = []

with open('./ark-tweet-nlp-0.3.2/output.txt','r') as f:
	file = f.readlines()

for lines in file[0:]:
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

	TagDict_list.append(TagDict)

with open('POS_result.csv', 'wb') as f:
	w = csv.DictWriter(f, fieldnames=TagDict_list[0].keys())
	w.writeheader()
	w.writerows(TagDict_list)

# Display all tags
#for one in TagDict_list:
#	print 'this is the tag count:',one
#	print '\n'
