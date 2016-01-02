# input the sentences file and output a pickle file, meanwhile display first three output
# Run by: python BrownCluster.py input.txt output.txt
import csv
import nltk
import pickle
import sys
# Load the Twitter Word Clusters into a dictionary
def get_cluster_dic(file):
	f = open(file)
	cluster_dic = {}
	csv_f = csv.reader(f, delimiter='\t')
	for row in csv_f:
		try:
			cluster_dic[row[1]].append(row[0])
		except KeyError:
			cluster_dic[row[1]] = [row[0]]
		except IndexError:
			break

	return cluster_dic

cluster_dic = get_cluster_dic("Twc.csv")

# Get 1000 clusters' names, file is the name of corpus
def get_cluster_name(file):
	f = open(file)
	cluster = []
	csv_f = csv.reader(f, delimiter='\t')	
	for row in csv_f:
		if row[0] in cluster:
			continue
		else:
			cluster.append(row[0])

	return cluster

# Given a sentence then tokenizer it into list of tokens, return a list of tuples of
# tokens which are in the Brown Cluster and a count of '1'.
def Map(L):
	L = nltk.word_tokenize(L)
	print "Sentence Tokenizers:", L
	results = []
	for word in L:
		if(cluster_dic.get(word)):
			results.append ((cluster_dic[word], 1))

	return results

"""
Group the sublists of (token, 1) pairs into a term-frequency-list
map, so that the Reduce operation later can work on sorted
term counts. The returned result is a dictionary with the structure
{token : [([token], 1), ...] .. }
"""
def Partition(L):

  	tf = {}
  	for p in L:
		try:
	    		tf[p[0][0]].append(p)
	  	except KeyError:
	    		tf[p[0][0]] = [p]

  	return tf


"""
Given a (token, [([token], 1) ...]) tuple, collapse all the
count tuples from the Map operation into a single term frequency
number for this token, and return a list of final tuple [(token, frequency),...].
"""
def Reduce(Mapping):
  return (Mapping[0], sum(pair[1] for pair in Mapping[1]))


if __name__ == '__main__':
	cluster_namelist = get_cluster_name('Twc.csv')
	try:
		arg1 = sys.argv[1]
		arg2 = sys.argv[2]
	except IndexError:
		print "You need to type input name and output name"
		sys.exit(1)

	outputlist = []
	inputfile = open(arg1, 'r')
	outputfile = open(arg2, 'w')
	count = 0
# Read the sentences in inputfile and process them by Brown Clustering
# Load the resultlist into outputfile and display first 3 results
	for sentence in inputfile.readlines():
		count = count + 1
		sentence_map = Map(sentence)
		sentence_reduce = map(Reduce, Partition(sentence_map).items()) 
		end_dic = {}
		end_list = []
		
		for cluster in cluster_namelist:
			end_dic[cluster] = 0

		for one in sentence_reduce:
			end_dic[one[0]] = one[1]

		for cluster in cluster_namelist:
			end_list.append(end_dic[cluster])

		if(count < 4):
			print "result", count, ":", sentence_reduce
			print "-"*50
		outputlist.append(end_list)

	pickle.dump(outputlist, outputfile)
	# To read it back: outputlist = pickle.load(outputfile)
	inputfile.close()
	outputfile.close()


