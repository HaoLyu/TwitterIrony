# input the sentences file and output a pickle file, meanwhile display first three output
# Run by: python BrownCluster.py input.txt output.txt
# It will also insert result into MongoDB
import csv
import nltk
import pickle
import sys
import re
from pymongo import MongoClient

# Load the Twitter Word Clusters into a dictionary
def get_cluster_dic(file="/Users/haolyu/TwitterIrony/irony_package/Brown Cluster/Twc.csv"):
	f = open(file)
	cluster_dic = {}
	cluster1000 = {}
	csv_f = csv.reader(f, delimiter='\t')
	for row in csv_f:
		if row[0] not in cluster1000:
			cluster1000[row[0]] = 1
		if row[1] in cluster_dic:
			cluster_dic[row[1]].append(row[0])
		else:
			cluster_dic[row[1]] = row[0]

	return (cluster_dic,cluster1000)

cluster_dic = get_cluster_dic("/Users/haolyu/TwitterIrony/irony_package/Brown Cluster/Twc.csv")[0]


# Given a sentence then tokenizer it into list of tokens, return a list of tuples of
# tokens which are in the Brown Cluster 
def Map(text):
	global cluster_dic
	text = re.sub(r"(?:\@|https?\://)\S+", "", text)
	pat = re.compile(r'\s+')
	text = pat.sub(' ', text).strip()
	L = nltk.word_tokenize(text)
	L = [w.lower() for w in L]
	results = []
	for word in L:
		if word in cluster_dic:
			results.append(cluster_dic[word])

	return results





if __name__ == '__main__':
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets
	number = dbtweets.find({'BrownCluster':{'$exists':False}}).count()
	for i in range(number):
		#try:
		tweet_id = dbtweets.find({'BrownCluster':{'$exists':False}})[0]['tweet_id']
		print tweet_id
		text = dbtweets.find({'BrownCluster':{'$exists':False}})[0]['tweet_text']
		BrownCluster = Map(text)
		
		result = dbtweets.update_one({"tweet_id": tweet_id},
			{
			    "$set": {
	                "BrownCluster":BrownCluster
	        	}
			}
		)
		
		#except Exception:
			#print i
			#continue
			

