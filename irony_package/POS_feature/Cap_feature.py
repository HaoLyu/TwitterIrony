# Captalization features: The number of word with initial caps and all caps and the number of POS tags with at least initial caps.
# The result will print in the terminal and stored in Cap_result.csv
import csv
TagList = ['N','O','S','^','Z','L','M','V','A','R',
		   '!','D','P','&','T','X','Y','#','@','~',
		   'U','E','$',',','G']

CapDict_list = []

with open('./ark-tweet-nlp-0.3.2/output.txt','r') as f:
	file = f.readlines()

for lines in file[0:]:
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

	CapDict_list.append(CapDict)
	
	
with open('Cap_result.csv', 'wb') as f:
	w = csv.DictWriter(f, fieldnames=CapDict_list[0].keys())
	w.writeheader()
	w.writerows(CapDict_list)