import BrownCluster as BC
import pickle
import csv
import numpy as np

# get the element wise product of the Brown Cluster scores of two sentences(target tweet and original one)
def pairwise_brown_score(target, origin):

	pairwise_BC = [target, origin]
	cluster_namelist = BC.get_cluster_name('Twc.csv')
	outputlist = []
		
	# Read the sentences in pairwise_BC and get the score of Brown Clustering
	for sentence in pairwise_BC:
		sentence_map = BC.Map(sentence)
		sentence_reduce = map(BC.Reduce, BC.Partition(sentence_map).items()) 
		end_dic = {}
		end_list = []

		for cluster in cluster_namelist:
			end_dic[cluster] = 0

		for one in sentence_reduce:
			end_dic[one[0]] = one[1]

		for cluster in cluster_namelist:
			end_list.append(end_dic[cluster])


		outputlist.append(end_list)

	result = list(np.multiply(outputlist[0], outputlist[1]))
	
	return (result, sum(result))


