# -*- coding: utf-8 -*-
# Code by Hao Lyu, UT Austin
# Run by: python Model_historical_feature.py Author_historical_salient_terms
import sys
sys.path.append('../irony_package/Brown Cluster')
sys.path.append('../irony_package')
import BrownCluster
import BrownCluster_original_tweet as BC_original_tweet
import Tweet_Transfer_BOW as BOW
import operator
import numpy as np
import lda
import pickle
import re
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn import grid_search
from sklearn import cross_validation
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn import preprocessing
import random
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

# Preprocess numerical feature using standardization
def preprocess_num(X_train):
	min_max_scaler = preprocessing.MinMaxScaler()
	X_train_minmax = min_max_scaler.fit_transform(X_train)
	return X_train_minmax

# Find the at(@) mentioned names in the list of tweets
def find_at_names_in_hist_tweets(hist_list):
	return_dict = {}
	audience = []
	for text in hist_list:
		audience = audience + re.findall( r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)', text)
	for j in range(len(audience)):
		audience[j] = audience[j].encode('utf-8').strip()
		if audience[j] not in return_dict:
			return_dict[audience[j]] = 1
		else:
			return_dict[audience[j]] += 1

	return return_dict

# Generate the perdicting accuracy using only top 100 tfidf terms
def Author_historical_salient_terms():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	tfidf_dict = {}
	fit_tfidf_list = []
	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'hist_list':{'$exists':True}}).count()): 
	#for i in range(1000):
		try:
			author_full_name = (dbtweets.find({'hist_list':{'$exists':True}})[i]['author_full_name']).encode('utf-8')
			tweetList = dbtweets.find({'hist_list':{'$exists':True}})[i]['hist_list']
			sarcasm_score = dbtweets.find({'hist_list':{'$exists':True}})[i]['sarcasm_score']
			top_tfidf_list = []
			vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,1), stop_words='english')
			tfidf_matrix = vectorizer.fit_transform(tweetList)
			idf = vectorizer.idf_
			scores =  dict(zip(vectorizer.get_feature_names(), idf))
			sortedList = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
			top_100_number = len(sortedList)
			if top_100_number < 100:
				for n in range(top_100_number):
					top_tfidf_list.append(sortedList[n][0])
			else:
				for n in range(100):
					top_tfidf_list.append(sortedList[n][0])
			
			for term in top_tfidf_list:
				if term not in tfidf_dict:
					tfidf_dict[term] = 1
				else:
					continue
			input_data = " ".join(top_tfidf_list)
			fit_tfidf_list.append(input_data)
			target_list.append(int(sarcasm_score.encode('utf-8')))
			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(fit_tfidf_list)-1)
			else:
				name_dict[author_full_name] = [len(fit_tfidf_list)-1]

		except ValueError:
			continue
	
	client.close()
	count_vect = CountVectorizer(vocabulary=tfidf_dict.keys(),binary=True)
	del tfidf_dict
	X = count_vect.fit_transform(fit_tfidf_list)
	del fit_tfidf_list
	#X = csr_matrix(X).toarray()
	Y_target = np.array(target_list)
	del target_list
	result = LR_model(name_dict, X, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])

# Generate the perdicting accuracy using only historical sentiment
def Author_historical_sentiment():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	sentiment_list = []
	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'hist_sentiment_neutral':{'$exists':True}}).count()): 
	#for i in xrange(1000):
		try:
			document = dbtweets.find({'hist_sentiment_neutral':{'$exists':True}})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			hist_sentiment_neutral = float(document['hist_sentiment_neutral'])
			hist_sentiment_positive = float(document['hist_sentiment_positive'])
			hist_sentiment_very_positive = float(document['hist_sentiment_very_positive'])
			hist_sentiment_negative = float(document['hist_sentiment_negative'])
			hist_sentiment_very_negative = float(document['hist_sentiment_very_negative'])
			sentiment_list.append([hist_sentiment_neutral, hist_sentiment_positive, hist_sentiment_very_positive, hist_sentiment_negative, hist_sentiment_very_negative])
			sarcasm_score = document['sarcasm_score']
			
			target_list.append(int(sarcasm_score.encode('utf-8')))
			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(sentiment_list)-1)
			else:
				name_dict[author_full_name] = [len(sentiment_list)-1]

		except ValueError:
			continue
	
	client.close()
	X = np.array(sentiment_list)
	del sentiment_list
	Y_target = np.array(target_list)
	del target_list
	result = LR_model(name_dict, X, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])


# Generate the perdicting accuracy using only bag of words
def word_unigrams_bigrams():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	unigrams_list = csr_matrix([])
	bigrams_list = csr_matrix([])
	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'word_unigrams':{'$exists':True}}).count()): 
	#for i in xrange(2000):
		try:
			document = dbtweets.find({'word_unigrams':{'$exists':True}})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			word_unigrams = pickle.loads(document['word_unigrams'])
			word_bigrams = pickle.loads(document['word_bigrams'])
			
			if unigrams_list.shape[1] == 0:
				unigrams_list = word_unigrams
			else:
				unigrams_list = vstack([unigrams_list,word_unigrams])

			if bigrams_list.shape[1] == 0:
				bigrams_list = word_bigrams
			else:
				bigrams_list = vstack([bigrams_list,word_bigrams])

			sarcasm_score = document['sarcasm_score']
			
			target_list.append(int(sarcasm_score.encode('utf-8')))
			if author_full_name in name_dict:
				name_dict[author_full_name].append(unigrams_list.shape[0]-1)
			else:
				name_dict[author_full_name] = [unigrams_list.shape[0]-1]
		except ValueError:
			continue
	
	client.close()
	Y_target = np.array(target_list)
	del target_list
	result = LR_model(name_dict, unigrams_list, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])
	result = LR_model(name_dict, bigrams_list, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])


# Generate the perdicting accuracy using only Brown Cluster unigrams and bigrams
def Brown_Cluster_unigrams_bigrams():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets
	bigram_dict = {}
	unigrams_list = []
	bigrams_list = []
	target_list = []
	name_dict = {}
	#for i in xrange(1000):
	for i in xrange(dbtweets.find().count()):
		document = dbtweets.find()[i]
		BC = document['BrownCluster']
		for j in range(len(BC)-1):
			bi_BC = (BC[j],BC[j+1])
			if bi_BC not in bigram_dict:
				bigram_dict[bi_BC] = 1


	bigram_dict_keys = bigram_dict.keys()
	for i in xrange(dbtweets.find().count()): 
	#for i in xrange(1000):
		try:
			document = dbtweets.find()[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			BC = document['BrownCluster']

			cluster1000_d = BrownCluster.get_cluster_dic()[1]
			cluster1000 = cluster1000_d.keys()

			word_unigrams = [0]*len(cluster1000)
			word_bigrams = [0]*len(bigram_dict_keys)
			for j in BC:
				word_unigrams[cluster1000.index(j)] = 1
			for j in range(len(BC)-1):
				bi_BC = (BC[j],BC[j+1])
				word_bigrams[bigram_dict_keys.index(bi_BC)] = 1

			
			unigrams_list.append(word_unigrams)
			bigrams_list.append(word_bigrams)
			sarcasm_score = document['sarcasm_score']
			
			target_list.append(int(sarcasm_score.encode('utf-8')))
			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(unigrams_list)-1)
			else:
				name_dict[author_full_name] = [len(unigrams_list)-1]
		except ValueError:
			continue
	
	client.close()
	X_unigrams = csr_matrix(np.array(unigrams_list))
	X_bigrams = csr_matrix(np.array(bigrams_list))
	Y_target = np.array(target_list)
	del target_list
	result = LR_model(name_dict, X_unigrams, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])
	result = LR_model(name_dict, X_bigrams, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])

# Generate the perdicting accuracy using only dependency arcs
def dependency_arcs():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	words_arc_dict = {}
	index = 0
	BC_arc_dict = {}
	BC_index = 0

	X_list_word_arc = []
	X_list_BC_arc = []
	X_word = []
	X_BC = []
	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'two_words_arc':{'$exists':True}}).count()): 
	#for i in xrange(1000):
		try:
			document = dbtweets.find({'two_words_arc':{'$exists':True}})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			two_words_arc_list = []
			two_BC_arc_list = []

			two_words_arc = document['two_words_arc']
			two_BC_arc = document['two_BC_arc']

			for one_arc in two_words_arc:
				if one_arc not in words_arc_dict:
					words_arc_dict[one_arc] = index
					index += 1
				two_words_arc_list.append(words_arc_dict[one_arc])

			X_list_word_arc.append(two_words_arc_list)

			for one_arc in two_BC_arc:
				if one_arc not in BC_arc_dict:
					BC_arc_dict[one_arc] = BC_index
					BC_index += 1
				two_BC_arc_list.append(BC_arc_dict[one_arc])

			X_list_BC_arc.append(two_BC_arc_list)

			sarcasm_score = document['sarcasm_score']
			
			target_list.append(int(sarcasm_score.encode('utf-8')))
			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(X_list_word_arc)-1)
			else:
				name_dict[author_full_name] = [len(X_list_word_arc)-1]

		except ValueError:
			continue
	
	words_arc_feature_len = len(words_arc_dict.keys())

	for one_list in X_list_word_arc:
		zero_list = [0]*words_arc_feature_len
		for j in one_list:
			zero_list[j] = 1
		X_word.append(zero_list)

	BC_arc_feature_len = len(BC_arc_dict.keys())

	for one_list in X_list_BC_arc:
		zero_list = [0]*BC_arc_feature_len
		for j in one_list:
			zero_list[j] = 1
		X_BC.append(zero_list)

	client.close()
	X_word = np.array(X_word)
	X_BC = np.array(X_BC)
	#X = np.concatenate((X_word, X_BC), axis=1)

	#del X_word, X_BC
	Y_target = np.array(target_list)
	del target_list
	result = LR_model(name_dict, X_word, Y_target)
	print 'Words arc has performance, accuracy is: %s, auc score is:%s'%(result[0], result[1])
	result = LR_model(name_dict, X_BC, Y_target)
	print 'Brown Cluster arc has performance, accuracy is: %s, auc score is:%s'%(result[0], result[1])

# Generate the perdicting accuracy using only intensifier
def intensifier():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	intensifier_list = []
	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'intensifier':{'$exists':True}}).count()): 
	#for i in xrange(1000):
		try:
			document = dbtweets.find({'intensifier':{'$exists':True}})[i]
			intensifier = document['intensifier']
			intensifier_list.append([intensifier])
			sarcasm_score = document['sarcasm_score']
			target_list.append(int(sarcasm_score.encode('utf-8')))
			author_full_name = document['author_full_name'].encode('utf-8')
			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(intensifier_list)-1)
			else:
				name_dict[author_full_name] = [len(intensifier_list)-1]

		except ValueError:
			continue
	
	client.close()
	X = np.array(intensifier_list)
	Y_target = np.array(target_list)
	del target_list, intensifier_list
	result = LR_model(name_dict, X, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])

# Generate the perdicting accuracy using only Part_of_speech feature
def Part_of_speech():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets
	TagList = ['N','O','S','^','Z','L','M','V','A','R',
	   			'!','D','P','&','T','X','Y','#','@','~',
				'U','E','$',',','G']
	Part_of_speech_list = []
	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'lexical_density':{'$exists':True}}).count()): 
	#for i in xrange(1000):
		try:
			row_feature = []
			document = dbtweets.find({'lexical_density':{'$exists':True}})[i]
			for a in TagList:			
				a_ratio = a+'_ratio'
				if a== '$':
					a = 'numeral_pos'
					a_ratio = a+'_ratio'
				a_value = document[a]
				a_ratio_value = document[a_ratio]
				row_feature.append(a_value)
				row_feature.append(a_ratio_value)
			row_feature.append(document['lexical_density'])
			Part_of_speech_list.append(row_feature)

			sarcasm_score = document['sarcasm_score']
			target_list.append(int(sarcasm_score.encode('utf-8')))
			author_full_name = document['author_full_name'].encode('utf-8')
			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(Part_of_speech_list)-1)
			else:
				name_dict[author_full_name] = [len(Part_of_speech_list)-1]

		except ValueError:
			continue
	
	client.close()
	X = np.array(Part_of_speech_list)
	Y_target = np.array(target_list)
	del target_list, Part_of_speech_list
	result = LR_model(name_dict, X, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])

# Generate the perdicting accuracy using only Capitalization feature
def Capitalization():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets
	TagList = ['N','O','S','^','Z','L','M','V','A','R',
	   			'!','D','P','&','T','X','Y','#','@','~',
				'U','E','$',',','G']
	Cap_list = []
	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'ini_cap_number':{'$exists':True}}).count()): 
	#for i in xrange(1000):
		try:
			row_feature = []
			document = dbtweets.find({'lexical_density':{'$exists':True}})[i]
			for a in TagList:			
				a_cap_count = a+'_cap_count'
				if a== '$':
					a_cap_count = 'numeral_pos_cap'
				a_value = document[a_cap_count]
				row_feature.append(a_value)

			row_feature.append(document['ini_cap_number'])
			row_feature.append(document['all_cap_number'])

			Cap_list.append(row_feature)

			sarcasm_score = document['sarcasm_score']
			target_list.append(int(sarcasm_score.encode('utf-8')))
			author_full_name = document['author_full_name'].encode('utf-8')
			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(Cap_list)-1)
			else:
				name_dict[author_full_name] = [len(Cap_list)-1]

		except ValueError:
			continue
	
	client.close()
	X = np.array(Cap_list)
	Y_target = np.array(target_list)
	del target_list, Cap_list
	result = LR_model(name_dict, X, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])

# Generate the perdicting accuracy using only Pronunciation feature
def Pronunciation():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	Pronunciation_list = []
	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'number_no_vowels':{'$exists':True}}).count()): 
		try:
			document = dbtweets.find({'intensifier':{'$exists':True}})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			Pronunciation_list.append([document["number_Polysyllables "],document["number_no_vowels"]])
			sarcasm_score = document['sarcasm_score']
			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(Pronunciation_list)-1)
			else:
				name_dict[author_full_name] = [len(Pronunciation_list)-1]

		except ValueError:
			continue
	
	client.close()
	X = np.array(Pronunciation_list)
	Y_target = np.array(target_list)
	del target_list, Pronunciation_list
	result = LR_model(name_dict, X, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])

# Generate the perdicting accuracy using only Tweet whole sentiment
def Tweet_whole_sentiment():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets
	senti_dict ={key:value for key, value in zip(["very negative", "negative", "neutral", "positive", "very positive"], range(5))}
	original_whole_sentiment = [0, 0, 0, 0, 0]
	Tweet_whole_sentiment_list = []
	target_list = []
	name_dict = {}
	#for i in xrange(1000):
	for i in xrange(dbtweets.find({'tweet_whole_sentimentpredict':{'$exists':True}}).count()): 
		try:
			document = dbtweets.find({'tweet_whole_sentimentpredict':{'$exists':True}})[i]
			this_sentence_sentiment = []
			author_full_name = document['author_full_name'].encode('utf-8')
			this_sentence_sentiment.append(float(document["positivenode"]))
			this_sentence_sentiment.append(float(document["negativenode"]))
			whole_sentiment = document['tweet_whole_sentimentpredict']
			original_whole_sentiment[senti_dict[whole_sentiment]] = 1
			this_sentence_sentiment = this_sentence_sentiment + original_whole_sentiment
			Tweet_whole_sentiment_list.append(this_sentence_sentiment)
			sarcasm_score = document['sarcasm_score']
			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(Tweet_whole_sentiment_list)-1)
			else:
				name_dict[author_full_name] = [len(Tweet_whole_sentiment_list)-1]

		except ValueError:
			continue
	
	client.close()
	X = np.array(Tweet_whole_sentiment_list)
	Y_target = np.array(target_list)
	del target_list, Tweet_whole_sentiment_list
	result = LR_model(name_dict, X, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])

# Generate the perdicting accuracy using only Tweet word sentiment
def Tweet_word_sentiment():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	Tweet_word_sentiment_list = []
	target_list = []
	name_dict = {}
	#for i in xrange(1000):
	for i in xrange(dbtweets.find({'effect_distance':{'$exists':True}}).count()): 
		try:
			document = dbtweets.find({'effect_distance':{'$exists':True}})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			Tweet_word_sentiment_list.append([document["effect_distance"],document["min_word_effect"],document["max_word_effect"],document["sentiment_distance"],document["min_word_senti"],document["max_word_senti"]])
			sarcasm_score = document['sarcasm_score']
			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(Tweet_word_sentiment_list)-1)
			else:
				name_dict[author_full_name] = [len(Tweet_word_sentiment_list)-1]

		except ValueError:
			continue
	
	client.close()
	X = np.array(Tweet_word_sentiment_list)
	Y_target = np.array(target_list)
	del target_list, Tweet_word_sentiment_list
	result = LR_model(name_dict, X, Y_target)
	#result = SVM_model(name_dict, X, Y_target)
	#result = LinearSVC_model(name_dict, X, Y_target)
	#result = GaussianNB_model(name_dict, X, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])

# Generate the perdicting accuracy using only Author_historical_topics
def Author_historical_topics():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	all_hist_list = []
	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'hist_list':{'$exists':True}}).count()): 
	#for i in xrange(1000):
		try:
			document = dbtweets.find({'hist_list':{'$exists':True}})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			hist_list = document['hist_list']
			len_of_hist_list = 0
			rand_order = random.sample(range(1, len(hist_list)), len(hist_list)-1)
			modified_hist_list = ''

			while(len_of_hist_list<1000):
				try:
					tweet_add = hist_list[rand_order.pop()].encode('utf-8')
					len_of_hist_list += len(tweet_add)
					modified_hist_list = modified_hist_list + '\n' + tweet_add
				except IndexError:
					break
			
			all_hist_list.append(modified_hist_list)
			sarcasm_score = document['sarcasm_score']			
			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(all_hist_list)-1)
			else:
				name_dict[author_full_name] = [len(all_hist_list)-1]
		except ValueError:
			continue
	
	client.close()
	X = BOW.Get_unigrams(all_hist_list)[1]
	Y = np.array(target_list)
	print type(X), type(Y)
	del target_list, all_hist_list

	name_list = name_dict.keys()
	total_set_length = len(name_list)
	kf = KFold(total_set_length, n_folds=5)
	avg_score = []
	avg_auc_score = []
	#print X.shape, Y.shape
	for train_index, test_index in kf:
		temp_index = []
		for i in train_index:
			temp_index += name_dict[name_list[i]]
		train_index = np.array(temp_index)

		temp_index = []
		for j in test_index:
			temp_index += name_dict[name_list[j]]
		test_index = np.array(temp_index)

		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		model = lda.LDA(n_topics=100, n_iter=1500, random_state=1)
		model.fit(X_train)
		X_train = model.doc_topic_
		X_test = model.transform(X_test)

		parameters = {'tol':[0.001,0.0001], 'C':[0.00001, 0.0001, 0.001, 0.1, 1, 10]}
		lr = linear_model.LogisticRegression(penalty='l2')
		clf = grid_search.GridSearchCV(lr, parameters)
		clf.fit(X_train, Y_train)
		best_params = clf.best_params_
		#del lr,clf
		#print best_params
		#tuned_clf = linear_model.LogisticRegression(penalty='l2', C=best_params['C'], tol=best_params['tol'])
		#tuned_clf.fit(X_train_part, Y_train_part)
		#y_true, y_pred = Y_test, tuned_clf.predict(X_test)
		#score = tuned_clf.score(X_test, Y_test)
		y_true, y_pred = Y_test, clf.predict(X_test)
		score = clf.score(X_test, Y_test)
		#del tuned_clf
		#print 'score is ',score
		avg_score.append(score)
		auc = roc_auc_score(y_true, y_pred)
		#print 'auc is', auc 
		avg_auc_score.append(auc)

	accuracy = reduce(lambda x, y: x + y, avg_score) / len(avg_score)
	auc_score = reduce(lambda x, y: x + y, avg_auc_score) / len(avg_auc_score)
	print 'accuracy is: %s, auc score is:%s'%(accuracy, auc_score)
	
# Generate the perdicting accuracy using only Author's profile information
def profile_information():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	profile_nofer_list = []
	profile_nofing_list = []
	profile_not_list = []
	profile_duration_list = []
	profile_avgt_list = []
	profile_verif_list = []
	profile_tz_list = []

	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({"time_zone":{"$exists":True}}).count()): 
	#for i in xrange(1000):
		try:
			document = dbtweets.find()[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			profile_nofer = document['followers_count']
			profile_nofing = document['following_count']
			profile_not = document['tweets_count']
			profile_duration = float(document['duration'])
			profile_avgt = document['avg_tweet']
			profile_verif = document['verified'].encode('utf-8')
			profile_tz = document['time_zone'].encode('utf-8')

			profile_nofer_list.append([profile_nofer])
			profile_nofing_list.append([profile_nofing])
			profile_not_list.append([profile_not])
			profile_duration_list.append([profile_duration])
			profile_avgt_list.append([profile_avgt])
			profile_verif_list.append(profile_verif)
			profile_tz_list.append(profile_tz)

			sarcasm_score = document['sarcasm_score']			
			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(profile_verif_list)-1)
			else:
				name_dict[author_full_name] = [len(profile_verif_list)-1]
		except ValueError:
			continue
		except KeyError:
			continue
	client.close()
	profile_nofer_list = preprocess_num(np.array(profile_nofer_list))
	profile_nofing_list = preprocess_num(np.array(profile_nofing_list))
	profile_not_list = preprocess_num(np.array(profile_not_list))
	profile_duration_list = preprocess_num(np.array(profile_duration_list))
	profile_avgt_list = preprocess_num(np.array(profile_avgt_list))
	count_vect = CountVectorizer()
	profile_verif_list = (count_vect.fit_transform(profile_verif_list)).toarray()
	count_vect = CountVectorizer()
	profile_tz_list = (count_vect.fit_transform(profile_tz_list)).toarray()
	
	X = np.concatenate((profile_nofer_list, profile_nofing_list, profile_not_list, 
						profile_duration_list, profile_avgt_list, profile_verif_list, 
						profile_tz_list), axis=1)
						
	X = csr_matrix(X)
	print X.shape
	Y_target = np.array(target_list)
	del target_list
	result = LR_model(name_dict, X, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])

# Generate the perdicting accuracy using only author historical sentiment
def author_historical_sentiment():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	sentiment_list = []
	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'hist_sentiment_positive':{'$exists':True}}).count()): 
	#for i in xrange(1000):
		try:
			document = dbtweets.find({'hist_sentiment_positive':{'$exists':True}})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			positive = float(document['hist_sentiment_positive'])
			very_positive = float(document['hist_sentiment_very_positive'])
			negative = float(document['hist_sentiment_negative'])
			very_negative = float(document['hist_sentiment_very_negative'])
			neutral = float(document['hist_sentiment_neutral'])
			
			sentiment_list.append([positive, very_positive, negative, very_negative, neutral])
			#sentiment_list.append([positive,negative,neutral])

			sarcasm_score = document['sarcasm_score']			
			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(sentiment_list)-1)
			else:
				name_dict[author_full_name] = [len(sentiment_list)-1]
		except ValueError:
			continue
	
	client.close()
	X = np.array(sentiment_list)
	Y_target = np.array(target_list)
	del target_list, sentiment_list
	result = LR_model(name_dict, X, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])



# Generate the perdicting accuracy using only the bag of words of profiles
def profile_unigrams():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	unigrams_list = csr_matrix([])
	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'profile_unigrams':{'$exists':True}}).count()): 
	#for i in xrange(1000):
		try:
			document = dbtweets.find({'profile_unigrams':{'$exists':True}})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			profile_unigrams = pickle.loads(document['profile_unigrams'])
			
			if unigrams_list.shape[1] == 0:
				unigrams_list = profile_unigrams
			else:
				unigrams_list = vstack([unigrams_list,profile_unigrams])

			sarcasm_score = document['sarcasm_score']			
			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(unigrams_list.shape[0]-1)
			else:
				name_dict[author_full_name] = [unigrams_list.shape[0]-1]
		except ValueError:
			continue
	
	client.close()
	Y_target = np.array(target_list)
	del target_list
	result = LR_model(name_dict, unigrams_list, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])

# Generate the perdicting accuracy using only Audience_historical_topics
def Audience_historical_topics():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	all_hist_list = []
	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'audience_hist_list':{'$exists':True}}).count()): 
	#for i in xrange(1000):
		try:
			document = dbtweets.find({'audience_hist_list':{'$exists':True}})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			hist_list = document['audience_hist_list']
			len_of_hist_list = 0
			rand_order = random.sample(range(1, len(hist_list)), len(hist_list)-1)
			modified_hist_list = ''

			while(len_of_hist_list<1000):
				try:
					tweet_add = hist_list[rand_order.pop()].encode('utf-8')
					len_of_hist_list += len(tweet_add)
					modified_hist_list = modified_hist_list + '\n' + tweet_add
				except IndexError:
					break
			
			all_hist_list.append(modified_hist_list)
			sarcasm_score = document['sarcasm_score']			
			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(all_hist_list)-1)
			else:
				name_dict[author_full_name] = [len(all_hist_list)-1]
		except ValueError:
			continue
	
	client.close()
	X = BOW.Get_unigrams(all_hist_list)[1]
	
	Y = np.array(target_list)
	print X.shape, Y.shape

	name_list = name_dict.keys()
	total_set_length = len(name_list)
	kf = KFold(total_set_length, n_folds=5)
	avg_score = []
	avg_auc_score = []
	#print X.shape, Y.shape
	for train_index, test_index in kf:
		temp_index = []
		for i in train_index:
			temp_index += name_dict[name_list[i]]
		train_index = np.array(temp_index)

		temp_index = []
		for j in test_index:
			temp_index += name_dict[name_list[j]]
		test_index = np.array(temp_index)

		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		model = lda.LDA(n_topics=100, n_iter=300, random_state=1)
		X_train = csr_matrix(X_train)
		model.fit(X_train)
		X_train = model.doc_topic_
		print X_train.shape
		X_test = model.transform(X_test)

		parameters = {'tol':[0.001,0.0001], 'C':[0.00001, 0.0001, 0.001, 0.1, 1, 10]}
		lr = linear_model.LogisticRegression(penalty='l2')
		clf = grid_search.GridSearchCV(lr, parameters)
		clf.fit(X_train, Y_train)
		y_true, y_pred = Y_test, clf.predict(X_test)
		score = clf.score(X_test, Y_test)
		avg_score.append(score)
		auc = roc_auc_score(y_true, y_pred)
		avg_auc_score.append(auc)

	accuracy = reduce(lambda x, y: x + y, avg_score) / len(avg_score)
	auc_score = reduce(lambda x, y: x + y, avg_auc_score) / len(avg_auc_score)
	print 'accuracy is: %s, auc score is:%s'%(accuracy, auc_score)

# Generate the perdicting accuracy using only top 100 tfidf terms in Audience historical tweets
def Audience_historical_salient_terms():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	tfidf_dict = {}
	fit_tfidf_list = []
	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'audience_hist_list':{'$exists':True}}).count()): 
	#for i in range(1000):
		try:
			author_full_name = (dbtweets.find({'audience_hist_list':{'$exists':True}})[i]['author_full_name']).encode('utf-8')
			tweetList = dbtweets.find({'audience_hist_list':{'$exists':True}})[i]['audience_hist_list']
			sarcasm_score = dbtweets.find({'audience_hist_list':{'$exists':True}})[i]['sarcasm_score']
			top_tfidf_list = []
			vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,1), stop_words='english')
			tfidf_matrix = vectorizer.fit_transform(tweetList)
			idf = vectorizer.idf_
			scores =  dict(zip(vectorizer.get_feature_names(), idf))
			sortedList = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
			top_100_number = len(sortedList)
			if top_100_number < 100:
				for n in range(top_100_number):
					top_tfidf_list.append(sortedList[n][0])
			else:
				for n in range(100):
					top_tfidf_list.append(sortedList[n][0])
			
			for term in top_tfidf_list:
				if term not in tfidf_dict:
					tfidf_dict[term] = 1
				else:
					continue
			input_data = " ".join(top_tfidf_list)
			fit_tfidf_list.append(input_data)
			target_list.append(int(sarcasm_score.encode('utf-8')))
			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(fit_tfidf_list)-1)
			else:
				name_dict[author_full_name] = [len(fit_tfidf_list)-1]

		except ValueError:
			continue
	
	client.close()
	count_vect = CountVectorizer(vocabulary=tfidf_dict.keys(),binary=True)
	X = count_vect.fit_transform(fit_tfidf_list)
	del fit_tfidf_list
	#X = csr_matrix(X).toarray()
	Y_target = np.array(target_list)
	del target_list
	result = LR_model(name_dict, X, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])

# Generate the perdicting accuracy using only the bag of words of Audiences' profiles
def Audience_profile_unigrams():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	unigrams_list = []
	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'audience_profile':{'$exists':True}}).count()): 
	#for i in xrange(1000):
		try:
			document = dbtweets.find({'audience_profile':{'$exists':True}})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			profile = document['audience_profile'].encode('utf-8')
			
			unigrams_list.append(profile)

			sarcasm_score = document['sarcasm_score']			
			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(unigrams_list)-1)
			else:
				name_dict[author_full_name] = [len(unigrams_list)-1]
		except ValueError:
			continue
	client.close()
	vectorizer = CountVectorizer(min_df=1, stop_words='english')
	X = vectorizer.fit_transform(unigrams_list)
	Y_target = np.array(target_list)
	del target_list
	result = LR_model(name_dict, X, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])

# Generate the perdicting accuracy using only Audience's profile information
def Audience_profile_information():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	profile_nofer_list = []
	profile_nofing_list = []
	profile_duration_list = []
	profile_avgt_list = []
	profile_veri_yes_list = []
	profile_veri_no_list = []
	profile_not_list = []

	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({"audience_following_count":{"$exists":True}}).count()): 
	#for i in xrange(1000):
		try:
			document = dbtweets.find()[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			profile_nofer = document['audience_followers_count']
			profile_nofing = document['audience_following_count']
			profile_duration = document['audience_duration']
			profile_avgt = document['audience_avg_tweet']
			profile_veri_yes = document['audience_verified_yes']
			profile_veri_no = document['audience_verified_no']
			profile_not = profile_avgt * profile_duration

			profile_nofer_list.append([profile_nofer])
			profile_nofing_list.append([profile_nofing])
			profile_not_list.append([profile_not])
			profile_duration_list.append([profile_duration])
			profile_avgt_list.append([profile_avgt])
			profile_veri_yes_list.append([profile_veri_yes])
			profile_veri_no_list.append([profile_veri_no])

			sarcasm_score = document['sarcasm_score']			
			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(profile_nofer_list)-1)
			else:
				name_dict[author_full_name] = [len(profile_nofer_list)-1]
		except ValueError:
			continue
		except KeyError:
			continue
	client.close()
	profile_nofer_list = preprocess_num(np.array(profile_nofer_list))
	profile_nofing_list = preprocess_num(np.array(profile_nofing_list))
	profile_not_list = preprocess_num(np.array(profile_not_list))
	profile_duration_list = preprocess_num(np.array(profile_duration_list))
	profile_avgt_list = preprocess_num(np.array(profile_avgt_list))
	profile_veri_yes_list = np.array(profile_veri_yes_list)
	profile_veri_no_list = np.array(profile_veri_no_list)

	X = np.concatenate((profile_nofer_list, profile_nofing_list, profile_duration_list,
						 profile_avgt_list, profile_veri_yes_list, profile_veri_no_list,
						profile_not_list), axis=1)
						
	X = csr_matrix(X)
	print X.shape
	Y_target = np.array(target_list)
	del target_list
	result = LR_model(name_dict, X, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])

# Generate the perdicting accuracy using only audience and author interactional topics
def interactional_topics():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	author_hist_list = []
	audience_hist_list = []
	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'$and':[{'audience_hist_list':{'$exists':True}},{'hist_list':{'$exists':True}}]}).count()): 
	#for i in xrange(1000):
		try:
			document = dbtweets.find({'$and':[{'audience_hist_list':{'$exists':True}},{'hist_list':{'$exists':True}}]})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			author_hist = document['hist_list']
			audience_hist = document['audience_hist_list']

			len_of_hist_list = 0
			rand_order = random.sample(range(1, len(author_hist)), len(author_hist)-1)
			modified_hist_list = ''

			while(len_of_hist_list<1000):
				try:
					tweet_add = author_hist[rand_order.pop()].encode('utf-8')
					len_of_hist_list += len(tweet_add)
					modified_hist_list = modified_hist_list +  tweet_add + '\n' 
				except IndexError:
					break
			
			author_hist_list.append(modified_hist_list)

			len_of_hist_list = 0
			rand_order = random.sample(range(1, len(audience_hist)), len(audience_hist)-1)
			modified_hist_list = ''

			while(len_of_hist_list<1000):
				try:
					tweet_add = audience_hist[rand_order.pop()].encode('utf-8')
					len_of_hist_list += len(tweet_add)
					modified_hist_list = modified_hist_list +  tweet_add + '\n' 
				except IndexError:
					break
			
			audience_hist_list.append(modified_hist_list)

			sarcasm_score = document['sarcasm_score']			
			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(author_hist_list)-1)
			else:
				name_dict[author_full_name] = [len(author_hist_list)-1]
		except ValueError:
			continue
	
	client.close()
	X_all = [a+b for a,b in zip(author_hist_list, audience_hist_list)]
	unicount_vect = CountVectorizer(ngram_range=(1,1), lowercase = True, stop_words='english', min_df=1)
	unicount_vect.fit(X_all)
	
	X_author = unicount_vect.transform(author_hist_list)
	X_audience = unicount_vect.transform(audience_hist_list)
	Y = np.array(target_list)
	print X_author.shape,X_audience.shape, Y.shape

	name_list = name_dict.keys()
	total_set_length = len(name_list)
	kf = KFold(total_set_length, n_folds=5)
	avg_score = []
	avg_auc_score = []
	#print X.shape, Y.shape
	for train_index, test_index in kf:
		temp_index = []
		for i in train_index:
			temp_index += name_dict[name_list[i]]
		train_index = np.array(temp_index)

		temp_index = []
		for j in test_index:
			temp_index += name_dict[name_list[j]]
		test_index = np.array(temp_index)
		X_author_train, X_author_test = X_author[train_index], X_author[test_index]
		X_audience_train, X_audience_test = X_audience[train_index], X_audience[test_index]
		X_author_train_index = X_author_train.shape[0]
		X_train = vstack([X_author_train, X_audience_train])

		model = lda.LDA(n_topics=100, n_iter=800, random_state=1)
		model.fit(X_train)
		doc_topic = model.doc_topic_
		X_author_train = doc_topic[:X_author_train_index]
		X_audience_train = doc_topic[X_author_train_index:]

		X_train = abs(X_author_train - X_audience_train)
		del X_author_train, X_audience_train
		min_max_scaler = preprocessing.MinMaxScaler()
		X_train = min_max_scaler.fit_transform(X_train)
		X_train = csr_matrix(X_train)
		X_author_test = X_author_test.toarray()
		X_audience_test = X_audience_test.toarray()
		X_author_test = model.transform(X_author_test)
		X_audience_test = model.transform(X_audience_test)

		X_test = abs(X_author_test - X_audience_test)
		del X_author_test,X_audience_test
		min_max_scaler = preprocessing.MinMaxScaler()
		X_test = min_max_scaler.fit_transform(X_test)
		X_test = csr_matrix(X_test)
		Y_train, Y_test = Y[train_index], Y[test_index]
		parameters = {'tol':[0.001,0.0001], 'C':[0.00001, 0.0001, 0.001, 0.1, 1, 10]}
		lr = linear_model.LogisticRegression(penalty='l2')
		clf = grid_search.GridSearchCV(lr, parameters)
		clf.fit(X_train, Y_train)
		y_true, y_pred = Y_test, clf.predict(X_test)
		score = clf.score(X_test, Y_test)
		avg_score.append(score)
		auc = roc_auc_score(y_true, y_pred)
		avg_auc_score.append(auc)
	accuracy = reduce(lambda x, y: x + y, avg_score) / len(avg_score)
	auc_score = reduce(lambda x, y: x + y, avg_auc_score) / len(avg_auc_score)
	print 'accuracy is: %s, auc score is:%s'%(accuracy, auc_score)
	


# Generate the perdicting accuracy using only historical communication between audience and author
def Audience_author_historical_communication():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	X = []
	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'$and':[{'audience_hist_list':{'$exists':True}},{'hist_list':{'$exists':True}}]}).count()): 
	#for i in range(1000):
		try:

			document = dbtweets.find({'$and':[{'audience_hist_list':{'$exists':True}},{'hist_list':{'$exists':True}}]})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			audience_full_name = document['audience']
			audience_hist_list = document['audience_hist_list']
			author_hist_list  = document['hist_list']
			sarcasm_score = document['sarcasm_score']

			author_at_name_dict = find_at_names_in_hist_tweets(author_hist_list)
			audience_at_name_dict = find_at_names_in_hist_tweets(audience_hist_list)

			message_sent = 0

			if author_full_name in audience_at_name_dict:
				message_sent = audience_at_name_dict[author_full_name]

			rank = 0.0

			if message_sent != 0:
				v = audience_at_name_dict.values()
				index = sorted(v).index(message_sent)
				rank = 	float(index + 1)/len(v) 

			mutual_meassage = 0

			if author_full_name in audience_at_name_dict:
				for one_aud in audience_full_name:
					if one_aud in author_at_name_dict:
						mutual_meassage = 1
						break

			X.append([message_sent, rank, mutual_meassage])
			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(X)-1)
			else:
				name_dict[author_full_name] = [len(X)-1]

		except ValueError:
			continue
	
	client.close()
	X = np.array(X)

	Y_target = np.array(target_list)
	del target_list
	result = LR_model(name_dict, X, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])

# Generate the perdicting accuracy using only pairwise Brown Cluster features between tweet and original message
def Pairwise_Brown_Cluster_original():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets
	X = np.array([])
	target_list = []
	name_dict = {}
	BC_values = {}
	BC_values = BC_original_tweet.get_cluster_dic()[1]

	temp_keys = BC_values.keys()
	length = len(temp_keys)
	for i in range(length):
		BC_values[temp_keys[i]] = i

	for i in xrange(dbtweets.find({'$and':[{'original_tweet_BrownCluster':{'$exists':True}},{'BrownCluster':{'$exists':True}}]}).count()): 
	#for i in xrange(1000):
		try:
			document = dbtweets.find({'$and':[{'original_tweet_BrownCluster':{'$exists':True}},{'BrownCluster':{'$exists':True}}]})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			BC = document['BrownCluster']
			original_BC = document['original_tweet_BrownCluster']


			pairwise_BC = [0]*length
			for bc in BC:
				temp_index = BC_values[bc]
				pairwise_BC[temp_index] = 1

			pairwise_OBC = [0]*length
			for obc in original_BC:
				temp_index = BC_values[obc]
				pairwise_OBC[temp_index] = 1

			pairwise_BC = np.array(pairwise_BC)
			pairwise_OBC = np.array(pairwise_OBC)
			#pairwise = pairwise_BC * pairwise_OBC
			pairwise = pairwise_BC | pairwise_OBC

			if X.shape[0]==0:
				X = pairwise
			else:
				X = np.vstack((X, pairwise))
			
			sarcasm_score = document['sarcasm_score']
			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(X)-1)
			else:
				name_dict[author_full_name] = [len(X)-1]
		except ValueError:
			continue
	
	client.close()
	X = csr_matrix(X)
	Y_target = np.array(target_list)
	del target_list
	result = LR_model(name_dict, X, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])


# Generate the perdicting accuracy using only the bag of words of original tweet
def original_tweet_unigrams():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	unigrams_list = []
	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'original_tweet':{'$exists':True}}).count()): 
	#for i in xrange(1000):
		try:
			document = dbtweets.find({'original_tweet':{'$exists':True}})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			original_tweet = document['original_tweet'].encode('utf-8')
			
			unigrams_list.append(original_tweet)

			sarcasm_score = document['sarcasm_score']			
			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(unigrams_list)-1)
			else:
				name_dict[author_full_name] = [len(unigrams_list)-1]
		except ValueError:
			continue
	client.close()
	vectorizer = CountVectorizer(min_df=1, stop_words='english',binary=True)
	X = vectorizer.fit_transform(unigrams_list)
	#print type(X)
	Y_target = np.array(target_list)
	del target_list
	result = LR_model(name_dict, X, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])


#SGD logloss
# Logistic Regression Model with 10 layer Cross Validation. 
# trainning set(8/10), parameter development set(1/10), test set(1/10).
def LR_model(name_dict, X, Y):
	print "X and Y's shape are",X.shape, Y.shape
	name_list = name_dict.keys()
	total_set_length = len(name_list)
	kf = KFold(total_set_length, n_folds=5)
	avg_score = []
	avg_auc_score = []
	for train_index, test_index in kf:
		temp_index = []
		for i in train_index:
			temp_index += name_dict[name_list[i]]
		train_index = np.array(temp_index)

		temp_index = []
		for j in test_index:
			temp_index += name_dict[name_list[j]]
		test_index = np.array(temp_index)
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]

		parameters = {'tol':[0.001,0.0001], 'C':[0.00001, 0.0001, 0.001, 0.1, 1, 10]}
		lr = linear_model.LogisticRegression(penalty='l2')
		clf = grid_search.GridSearchCV(lr, parameters)
		clf.fit(X_train, Y_train)
		best_params = clf.best_params_
		y_true, y_pred = Y_test, clf.predict(X_test)
		score = clf.score(X_test, Y_test)
		avg_score.append(score)
		auc = roc_auc_score(y_true, y_pred)
		avg_auc_score.append(auc)

	accuracy = reduce(lambda x, y: x + y, avg_score) / len(avg_score)
	auc_score = reduce(lambda x, y: x + y, avg_auc_score) / len(avg_auc_score)
	return (accuracy,auc_score)

def SVM_model(name_dict, X, Y):
	print "X and Y's shape are",X.shape, Y.shape
	name_list = name_dict.keys()
	total_set_length = len(name_list)
	kf = KFold(total_set_length, n_folds=5)
	avg_score = []
	avg_auc_score = []
	for train_index, test_index in kf:
		temp_index = []
		for i in train_index:
			temp_index += name_dict[name_list[i]]
		train_index = np.array(temp_index)

		temp_index = []
		for j in test_index:
			temp_index += name_dict[name_list[j]]
		test_index = np.array(temp_index)
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]

		parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
					{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
		svc = svm.SVC(C=1)
		clf = grid_search.GridSearchCV(svc, parameters)
		clf.fit(X_train, Y_train)
		y_true, y_pred = Y_test, clf.predict(X_test)
		score = clf.score(X_test, Y_test)
		avg_score.append(score)
		auc = roc_auc_score(y_true, y_pred)
		avg_auc_score.append(auc)

	accuracy = reduce(lambda x, y: x + y, avg_score) / len(avg_score)
	auc_score = reduce(lambda x, y: x + y, avg_auc_score) / len(avg_auc_score)
	return (accuracy,auc_score)

def LinearSVC_model(name_dict, X, Y):
	print "X and Y's shape are",X.shape, Y.shape
	name_list = name_dict.keys()
	total_set_length = len(name_list)
	kf = KFold(total_set_length, n_folds=5)
	avg_score = []
	avg_auc_score = []
	for train_index, test_index in kf:
		temp_index = []
		for i in train_index:
			temp_index += name_dict[name_list[i]]
		train_index = np.array(temp_index)

		temp_index = []
		for j in test_index:
			temp_index += name_dict[name_list[j]]
		test_index = np.array(temp_index)
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]

		parameters = {'tol':[0.001,0.0001], 'C':[0.00001, 0.0001, 0.001, 0.1, 1, 10]}
		svc = svm.LinearSVC(penalty='l2')
		clf = grid_search.GridSearchCV(svc, parameters)
		clf.fit(X_train, Y_train)
		best_params = clf.best_params_
		y_true, y_pred = Y_test, clf.predict(X_test)
		score = clf.score(X_test, Y_test)
		avg_score.append(score)
		auc = roc_auc_score(y_true, y_pred)
		avg_auc_score.append(auc)

	accuracy = reduce(lambda x, y: x + y, avg_score) / len(avg_score)
	auc_score = reduce(lambda x, y: x + y, avg_auc_score) / len(avg_auc_score)
	return (accuracy,auc_score)

def GaussianNB_model(name_dict, X, Y):
	print "X and Y's shape are",X.shape, Y.shape
	name_list = name_dict.keys()
	total_set_length = len(name_list)
	kf = KFold(total_set_length, n_folds=5)
	avg_score = []
	avg_auc_score = []
	for train_index, test_index in kf:
		temp_index = []
		for i in train_index:
			temp_index += name_dict[name_list[i]]
		train_index = np.array(temp_index)

		temp_index = []
		for j in test_index:
			temp_index += name_dict[name_list[j]]
		test_index = np.array(temp_index)
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]

		parameters = {'tol':[0.001,0.0001], 'C':[0.00001, 0.0001, 0.001, 0.1, 1, 10]}
		clf = GaussianNB()
		clf.fit(X_train, Y_train)
		y_true, y_pred = Y_test, clf.predict(X_test)
		score = clf.score(X_test, Y_test)
		avg_score.append(score)
		auc = roc_auc_score(y_true, y_pred)
		avg_auc_score.append(auc)

	accuracy = reduce(lambda x, y: x + y, avg_score) / len(avg_score)
	auc_score = reduce(lambda x, y: x + y, avg_auc_score) / len(avg_auc_score)
	return (accuracy,auc_score)


##### larger feature combinations

# Generate the perdicting accuracy using Environment Features
def environment_features():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets
	X = np.array([])
	unigrams_list = []
	target_list = []
	name_dict = {}
	BC_values = BC_original_tweet.get_cluster_dic()[1]

	temp_keys = BC_values.keys()
	length = len(temp_keys)
	for i in range(length):
		BC_values[temp_keys[i]] = i

	for i in xrange(dbtweets.find({'environment_features':{'$exists':True}}).count()):
	#for i in xrange(1000):
		try:
			document = dbtweets.find({'environment_features':{'$exists':True}})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			original_tweet = document['original_tweet'].encode('utf-8')
			BC = document['BrownCluster']
			original_BC = document['original_tweet_BrownCluster']

			unigrams_list.append(original_tweet)

			pairwise_BC = [0]*length
			for bc in BC:
				temp_index = BC_values[bc]
				pairwise_BC[temp_index] = 1

			pairwise_OBC = [0]*length
			for obc in original_BC:
				temp_index = BC_values[obc]
				pairwise_OBC[temp_index] = 1

			pairwise_BC = np.array(pairwise_BC)
			pairwise_OBC = np.array(pairwise_OBC)
			#pairwise = pairwise_BC * pairwise_OBC
			pairwise = pairwise_BC | pairwise_OBC

			if X.shape[0]==0:
				X = pairwise
			else:
				X = np.vstack((X, pairwise))
			
			sarcasm_score = document['sarcasm_score']
			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(X)-1)
			else:
				name_dict[author_full_name] = [len(X)-1]
		except ValueError:
			continue
	
	client.close()
	vectorizer = CountVectorizer(min_df=1, stop_words='english',binary=True)
	X_uni = vectorizer.fit_transform(unigrams_list)
	#print "X_uni.shape",X_uni.shape
	#print type(X_uni)
	X = csr_matrix(X)
	#print "X shape",X.shape
	X = hstack([X, X_uni]).toarray()

	Y_target = np.array(target_list)
	del target_list
	result = LR_model(name_dict, X, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])

# Generate the perdicting accuracy using Audience Features
def audience_features():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	X = []
	tfidf_dict = {}
	fit_tfidf_list = []
	unigrams_list = []
	author_hist_list = []
	audience_hist_list = []
	profile_nofer_list = []
	profile_nofing_list = []
	profile_duration_list = []
	profile_avgt_list = []
	profile_veri_yes_list = []
	profile_veri_no_list = []
	profile_not_list = []


	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'audience_features':{'$exists':True}}).count()):
	#for i in range(2000):
		try:

			document = dbtweets.find({'audience_features':{'$exists':True}})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			audience_full_name = document['audience']
			audience_hist = document['audience_hist_list']
			author_hist  = document['hist_list']
			sarcasm_score = document['sarcasm_score']
			profile_nofer = document['audience_followers_count']
			profile_nofing = document['audience_following_count']
			profile_duration = document['audience_duration']
			profile_avgt = document['audience_avg_tweet']
			profile_veri_yes = document['audience_verified_yes']
			profile_veri_no = document['audience_verified_no']
			profile_not = profile_avgt * profile_duration
			profile = document['audience_profile'].encode('utf-8')

			unigrams_list.append(profile)
			profile_nofer_list.append([profile_nofer])
			profile_nofing_list.append([profile_nofing])
			profile_not_list.append([profile_not])
			profile_duration_list.append([profile_duration])
			profile_avgt_list.append([profile_avgt])
			profile_veri_yes_list.append([profile_veri_yes])
			profile_veri_no_list.append([profile_veri_no])

			author_at_name_dict = find_at_names_in_hist_tweets(author_hist)
			audience_at_name_dict = find_at_names_in_hist_tweets(audience_hist)

			message_sent = 0

			if author_full_name in audience_at_name_dict:
				message_sent = audience_at_name_dict[author_full_name]

			rank = 0.0

			if message_sent != 0:
				v = audience_at_name_dict.values()
				index = sorted(v).index(message_sent)
				rank = 	float(index + 1)/len(v) 

			mutual_meassage = 0

			if author_full_name in audience_at_name_dict:
				for one_aud in audience_full_name:
					if one_aud in author_at_name_dict:
						mutual_meassage = 1
						break

			X.append([message_sent, rank, mutual_meassage])

			len_of_hist_list = 0
			rand_order = random.sample(range(1, len(author_hist)), len(author_hist)-1)
			modified_hist_list = ''

			while(len_of_hist_list<1000):
				try:
					tweet_add = author_hist[rand_order.pop()].encode('utf-8')
					len_of_hist_list += len(tweet_add)
					modified_hist_list = modified_hist_list +  tweet_add + '\n' 
				except IndexError:
					break
			
			author_hist_list.append(modified_hist_list)

			len_of_hist_list = 0
			rand_order = random.sample(range(1, len(audience_hist)), len(audience_hist)-1)
			modified_hist_list = ''

			while(len_of_hist_list<1000):
				try:
					tweet_add = audience_hist[rand_order.pop()].encode('utf-8')
					len_of_hist_list += len(tweet_add)
					modified_hist_list = modified_hist_list +  tweet_add + '\n' 
				except IndexError:
					break
			
			audience_hist_list.append(modified_hist_list)

			top_tfidf_list = []
			vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,1), stop_words='english')
			tfidf_matrix = vectorizer.fit_transform(audience_hist)
			idf = vectorizer.idf_
			scores =  dict(zip(vectorizer.get_feature_names(), idf))
			sortedList = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
			top_100_number = len(sortedList)
			if top_100_number < 100:
				for n in range(top_100_number):
					top_tfidf_list.append(sortedList[n][0])
			else:
				for n in range(100):
					top_tfidf_list.append(sortedList[n][0])
			
			for term in top_tfidf_list:
				if term not in tfidf_dict:
					tfidf_dict[term] = 1
				else:
					continue
			input_data = " ".join(top_tfidf_list)
			fit_tfidf_list.append(input_data)




			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(X)-1)
			else:
				name_dict[author_full_name] = [len(X)-1]

		except ValueError:
			continue
	
	client.close()
	X_aud_hist_topics = BOW.Get_unigrams(audience_hist_list)[1]
	X_all = [a+b for a,b in zip(author_hist_list, audience_hist_list)]
	unicount_vect = CountVectorizer(ngram_range=(1,1), lowercase = True, stop_words='english', min_df=1)
	unicount_vect.fit(X_all)
	X_author = unicount_vect.transform(author_hist_list)
	X_audience = unicount_vect.transform(audience_hist_list)

	vectorizer = CountVectorizer(min_df=1, stop_words='english')
	X_unigrams = vectorizer.fit_transform(unigrams_list)

	count_vect = CountVectorizer(vocabulary=tfidf_dict.keys(),binary=True)
	X_tfidf = count_vect.fit_transform(fit_tfidf_list)



	profile_nofer_list = preprocess_num(np.array(profile_nofer_list))
	profile_nofing_list = preprocess_num(np.array(profile_nofing_list))
	profile_not_list = preprocess_num(np.array(profile_not_list))
	profile_duration_list = preprocess_num(np.array(profile_duration_list))
	profile_avgt_list = preprocess_num(np.array(profile_avgt_list))
	profile_veri_yes_list = np.array(profile_veri_yes_list)
	profile_veri_no_list = np.array(profile_veri_no_list)

	X_profile_info = np.concatenate((profile_nofer_list, profile_nofing_list, profile_duration_list,
						 profile_avgt_list, profile_veri_yes_list, profile_veri_no_list,
						profile_not_list), axis=1)
	X = np.hstack((np.array(X),X_profile_info))
	X_his_com = csr_matrix(X)
	X_his_com = hstack([X_his_com, X_tfidf, X_unigrams])
	X_his_com = csr_matrix(X_his_com)

	
	

	Y= np.array(target_list)
	del target_list
	print X_author.shape,X_audience.shape, X_his_com.shape, Y.shape
	name_list = name_dict.keys()
	total_set_length = len(name_list)
	kf = KFold(total_set_length, n_folds=5)
	avg_score = []
	avg_auc_score = []
	#print X.shape, Y.shape
	for train_index, test_index in kf:
		temp_index = []
		for i in train_index:
			temp_index += name_dict[name_list[i]]
		train_index = np.array(temp_index)

		temp_index = []
		for j in test_index:
			temp_index += name_dict[name_list[j]]
		test_index = np.array(temp_index)
		X_author_train, X_author_test = X_author[train_index], X_author[test_index]
		X_audience_train, X_audience_test = X_audience[train_index], X_audience[test_index]

		X_his_com_train, X_his_com_test = X_his_com[train_index], X_his_com[test_index]
		X_his_com_train = csr_matrix(X_his_com_train)
		X_his_com_test = csr_matrix(X_his_com_test)
		X_aud_hist_topics_train, X_aud_hist_topics_test = X_aud_hist_topics[train_index], X_aud_hist_topics[test_index]
		model = lda.LDA(n_topics=100, n_iter=1500, random_state=1)
		X_aud_hist_topics_train = csr_matrix(X_aud_hist_topics_train)
		model.fit(X_aud_hist_topics_train)
		X_aud_hist_topics_train = model.doc_topic_
		X_aud_hist_topics_train = csr_matrix(X_aud_hist_topics_train)
		X_aud_hist_topics_test = model.transform(X_aud_hist_topics_test)
		X_aud_hist_topics_test = csr_matrix(X_aud_hist_topics_test)

		X_author_train_index = X_author_train.shape[0]
		X_train = vstack([X_author_train, X_audience_train])
		model = lda.LDA(n_topics=100, n_iter=1500, random_state=1)
		model.fit(X_train)
		doc_topic = model.doc_topic_
		X_author_train = doc_topic[:X_author_train_index]
		X_audience_train = doc_topic[X_author_train_index:]

		X_train = abs(X_author_train - X_audience_train)
		del X_author_train, X_audience_train
		min_max_scaler = preprocessing.MinMaxScaler()
		X_train = min_max_scaler.fit_transform(X_train)
		X_train = csr_matrix(X_train)
		X_train = hstack([X_train, X_his_com_train, X_aud_hist_topics_train])

		X_author_test = X_author_test.toarray()
		X_audience_test = X_audience_test.toarray()
		X_author_test = model.transform(X_author_test)
		X_audience_test = model.transform(X_audience_test)
		X_test = abs(X_author_test - X_audience_test)
		del X_author_test,X_audience_test
		min_max_scaler = preprocessing.MinMaxScaler()
		X_test = min_max_scaler.fit_transform(X_test)
		X_test = csr_matrix(X_test)
		X_test = hstack([X_test, X_his_com_test, X_aud_hist_topics_test])

		Y_train, Y_test = Y[train_index], Y[test_index]
		parameters = {'tol':[0.001,0.0001], 'C':[0.00001, 0.0001, 0.001, 0.1, 1, 10]}
		lr = linear_model.LogisticRegression(penalty='l2')
		clf = grid_search.GridSearchCV(lr, parameters)
		clf.fit(X_train, Y_train)
		y_true, y_pred = Y_test, clf.predict(X_test)
		score = clf.score(X_test, Y_test)
		avg_score.append(score)
		auc = roc_auc_score(y_true, y_pred)
		avg_auc_score.append(auc)
	accuracy = reduce(lambda x, y: x + y, avg_score) / len(avg_score)
	auc_score = reduce(lambda x, y: x + y, avg_auc_score) / len(avg_auc_score)
	print 'accuracy is: %s, auc score is:%s'%(accuracy, auc_score)

# Generate the perdicting accuracy using Author Features
def author_features():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	all_hist_list = []
	tfidf_dict = {}
	fit_tfidf_list = []
	profile_nofer_list = []
	profile_nofing_list = []
	profile_not_list = []
	profile_duration_list = []
	profile_avgt_list = []
	profile_verif_list = []
	profile_tz_list = []
	sentiment_list = []
	unigrams_list = []

	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'author_features':{'$exists':True}}).count()): 
	#for i in xrange(1000):
		try:
			document = dbtweets.find({'author_features':{'$exists':True}})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			hist_list = document['hist_list']
			profile_nofer = document['followers_count']
			profile_nofing = document['following_count']
			profile_not = document['tweets_count']
			profile_duration = float(document['duration'])
			profile_avgt = document['avg_tweet']
			profile_verif = document['verified'].encode('utf-8')
			profile_tz = document['time_zone'].encode('utf-8')
			positive = float(document['hist_sentiment_positive'])
			very_positive = float(document['hist_sentiment_very_positive'])
			negative = float(document['hist_sentiment_negative'])
			very_negative = float(document['hist_sentiment_very_negative'])
			neutral = float(document['hist_sentiment_neutral'])
			profile_unigrams = document['profile'].encode('utf-8')

			len_of_hist_list = 0
			rand_order = random.sample(range(1, len(hist_list)), len(hist_list)-1)
			modified_hist_list = ''

			while(len_of_hist_list<1000):
				try:
					tweet_add = hist_list[rand_order.pop()].encode('utf-8')
					len_of_hist_list += len(tweet_add)
					modified_hist_list = modified_hist_list + '\n' + tweet_add
				except IndexError:
					break
			
			all_hist_list.append(modified_hist_list)

			top_tfidf_list = []
			vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,1), stop_words='english')
			tfidf_matrix = vectorizer.fit_transform(hist_list)
			idf = vectorizer.idf_
			scores =  dict(zip(vectorizer.get_feature_names(), idf))
			sortedList = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
			top_100_number = len(sortedList)
			if top_100_number < 100:
				for n in range(top_100_number):
					top_tfidf_list.append(sortedList[n][0])
			else:
				for n in range(100):
					top_tfidf_list.append(sortedList[n][0])
			
			for term in top_tfidf_list:
				if term not in tfidf_dict:
					tfidf_dict[term] = 1
				else:
					continue
			input_data = " ".join(top_tfidf_list)
			fit_tfidf_list.append(input_data)

			profile_nofer_list.append([profile_nofer])
			profile_nofing_list.append([profile_nofing])
			profile_not_list.append([profile_not])
			profile_duration_list.append([profile_duration])
			profile_avgt_list.append([profile_avgt])
			profile_verif_list.append(profile_verif)
			profile_tz_list.append(profile_tz)
			sentiment_list.append([positive, very_positive, negative, very_negative, neutral])
			
			unigrams_list.append(profile_unigrams)
			

			sarcasm_score = document['sarcasm_score']			
			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(all_hist_list)-1)
			else:
				name_dict[author_full_name] = [len(all_hist_list)-1]
		except ValueError:
			continue

	client.close()
	X = BOW.Get_unigrams(all_hist_list)[1]
	count_vect = CountVectorizer(vocabulary=tfidf_dict.keys(),binary=True)
	del tfidf_dict
	X_salient_terms = count_vect.fit_transform(fit_tfidf_list)
	del fit_tfidf_list
	profile_nofer_list = preprocess_num(np.array(profile_nofer_list))
	profile_nofing_list = preprocess_num(np.array(profile_nofing_list))
	profile_not_list = preprocess_num(np.array(profile_not_list))
	profile_duration_list = preprocess_num(np.array(profile_duration_list))
	profile_avgt_list = preprocess_num(np.array(profile_avgt_list))
	count_vect = CountVectorizer()
	profile_verif_list = (count_vect.fit_transform(profile_verif_list)).toarray()
	count_vect = CountVectorizer()
	profile_tz_list = (count_vect.fit_transform(profile_tz_list)).toarray()
	X_profile_info = np.concatenate((profile_nofer_list, profile_nofing_list, profile_not_list, 
						profile_duration_list, profile_avgt_list, profile_verif_list, 
						profile_tz_list), axis=1)
	X_profile_info = csr_matrix(X_profile_info)
	X_sentiment = csr_matrix(np.array(sentiment_list))
	count_vect = CountVectorizer(stop_words='english',binary=True)
	unigrams_list = count_vect.fit_transform(unigrams_list)
	X_other = hstack([X_profile_info, X_salient_terms, X_sentiment, unigrams_list])
	X_other = csr_matrix(X_other)
	del X_profile_info, X_salient_terms, X_sentiment, unigrams_list
	Y = np.array(target_list)
	del target_list, all_hist_list
	name_list = name_dict.keys()
	total_set_length = len(name_list)
	kf = KFold(total_set_length, n_folds=5)
	avg_score = []
	avg_auc_score = []
	for train_index, test_index in kf:
		temp_index = []
		for i in train_index:
			temp_index += name_dict[name_list[i]]
		train_index = np.array(temp_index)

		temp_index = []
		for j in test_index:
			temp_index += name_dict[name_list[j]]
		test_index = np.array(temp_index)
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		X_other_train, X_other_test = X_other[train_index], X_other[test_index]
		model = lda.LDA(n_topics=100, n_iter=1500, random_state=1)
		X_train = csr_matrix(X_train)
		X_test = csr_matrix(X_test)
		model.fit(X_train)
		X_train = model.doc_topic_
		X_train = hstack([X_train, X_other_train])
		X_test = X_test.toarray()
		X_test = model.transform(X_test)
		X_test = csr_matrix(X_test)
		X_test = hstack([X_test, X_other_test])
		parameters = {'tol':[0.001,0.0001], 'C':[0.00001, 0.0001, 0.001, 0.1, 1, 10]}
		lr = linear_model.LogisticRegression(penalty='l2')
		clf = grid_search.GridSearchCV(lr, parameters)
		clf.fit(X_train, Y_train)
		y_true, y_pred = Y_test, clf.predict(X_test)
		score = clf.score(X_test, Y_test)
		avg_score.append(score)
		auc = roc_auc_score(y_true, y_pred)
		avg_auc_score.append(auc)
		print score, auc

	accuracy = reduce(lambda x, y: x + y, avg_score) / len(avg_score)
	auc_score = reduce(lambda x, y: x + y, avg_auc_score) / len(avg_auc_score)
	print 'accuracy is: %s, auc score is:%s'%(accuracy, auc_score)

# Generate the perdicting accuracy using Tweet Features
def tweet_features():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	unigrams_list = []
	bigrams_list = []

	bigram_dict = {}
	bcunigrams_list = []
	bcbigrams_list = []

	words_arc_dict = {}
	index = 0
	BC_arc_dict = {}
	BC_index = 0
	X_list_word_arc = []
	X_list_BC_arc = []
	X_word = []
	X_BC = []

	TagList = ['N','O','S','^','Z','L','M','V','A','R',
	   			'!','D','P','&','T','X','Y','#','@','~',
				'U','E','$',',','G']
	Part_of_speech_list = []
	Pronunciation_list = []
	Cap_list = []
	senti_dict ={key:value for key, value in zip(["very negative", "negative", "neutral", "positive", "very positive"], range(5))}
	original_whole_sentiment = [0, 0, 0, 0, 0]
	Tweet_whole_sentiment_list = []
	Tweet_word_sentiment_list = []
	intensifier_list = []

	target_list = []
	name_dict = {}
	for i in xrange(dbtweets.find({'tweet_features':{'$exists':True}}).count()): 
	#for i in xrange(500):
		document = dbtweets.find({'tweet_features':{'$exists':True}})[i]
		BC = document['BrownCluster']
		for j in range(len(BC)-1):
			bi_BC = (BC[j],BC[j+1])
			if bi_BC not in bigram_dict:
				bigram_dict[bi_BC] = 1

	bigram_dict_keys = bigram_dict.keys()
	for i in xrange(dbtweets.find({'tweet_features':{'$exists':True}}).count()):  
	#for i in xrange(500):
		document = dbtweets.find({'tweet_features':{'$exists':True}})[i]
		author_full_name = document['author_full_name'].encode('utf-8')
		word_unigrams = document['tweet_text'].encode('utf-8')

		BC = document['BrownCluster']

		Pronunciation_list.append([document["number_Polysyllables "],document["number_no_vowels"]])
		
		unigrams_list.append(word_unigrams)

		bigrams_list.append(word_unigrams)

		cluster1000_d = BrownCluster.get_cluster_dic()[1]
		cluster1000 = cluster1000_d.keys()
		bcword_unigrams = [0]*len(cluster1000)
		bcword_bigrams = [0]*len(bigram_dict_keys)
		for j in BC:
			bcword_unigrams[cluster1000.index(j)] = 1
		for j in range(len(BC)-1):
			bi_BC = (BC[j],BC[j+1])
			bcword_bigrams[bigram_dict_keys.index(bi_BC)] = 1

		bcunigrams_list.append(bcword_unigrams)
		bcbigrams_list.append(bcword_bigrams)

		two_words_arc_list = []
		two_BC_arc_list = []

		two_words_arc = document['two_words_arc']
		two_BC_arc = document['two_BC_arc']

		for one_arc in two_words_arc:
			if one_arc not in words_arc_dict:
				words_arc_dict[one_arc] = index
				index += 1
			two_words_arc_list.append(words_arc_dict[one_arc])
		X_list_word_arc.append(two_words_arc_list)

		for one_arc in two_BC_arc:
			if one_arc not in BC_arc_dict:
				BC_arc_dict[one_arc] = BC_index
				BC_index += 1
			two_BC_arc_list.append(BC_arc_dict[one_arc])
		X_list_BC_arc.append(two_BC_arc_list)

		row_feature = []
		for a in TagList:			
			a_ratio = a+'_ratio'
			if a== '$':
				a = 'numeral_pos'
				a_ratio = a+'_ratio'
			a_value = document[a]
			a_ratio_value = document[a_ratio]
			row_feature.append(a_value)
			row_feature.append(a_ratio_value)
		row_feature.append(document['lexical_density'])
		Part_of_speech_list.append(row_feature)

		cap_row_feature = []
		for a in TagList:			
			a_cap_count = a+'_cap_count'
			if a== '$':
				a_cap_count = 'numeral_pos_cap'
			a_value = document[a_cap_count]
			cap_row_feature.append(a_value)

		cap_row_feature.append(document['ini_cap_number'])
		cap_row_feature.append(document['all_cap_number'])

		Cap_list.append(cap_row_feature)
		this_sentence_sentiment = []
		this_sentence_sentiment.append(float(document["positivenode"]))
		this_sentence_sentiment.append(float(document["negativenode"]))
		whole_sentiment = document['tweet_whole_sentimentpredict']
		original_whole_sentiment[senti_dict[whole_sentiment]] = 1
		this_sentence_sentiment = this_sentence_sentiment + original_whole_sentiment
		Tweet_whole_sentiment_list.append(this_sentence_sentiment)
		Tweet_word_sentiment_list.append([document["effect_distance"],document["min_word_effect"],document["max_word_effect"],document["sentiment_distance"],document["min_word_senti"],document["max_word_senti"]])
		intensifier = document['intensifier']
		intensifier_list.append([intensifier])

		sarcasm_score = document['sarcasm_score']			
		target_list.append(int(sarcasm_score.encode('utf-8')))
		if author_full_name in name_dict:
			name_dict[author_full_name].append(len(target_list)-1)
		else:
			name_dict[author_full_name] = [len(target_list)-1]
		#except ValueError:
			#continue
	client.close()
	unicount_vect = CountVectorizer(ngram_range=(1,1), lowercase = True,  stop_words='english', min_df=1)
	X_unigrams = unicount_vect.fit_transform(unigrams_list)
	bicount_vect = CountVectorizer(ngram_range=(2,2), lowercase = True, stop_words='english',  min_df=1)
	X_bigrams = bicount_vect.fit_transform(bigrams_list)

	X_bcunigrams = csr_matrix(np.array(bcunigrams_list))
	X_bcbigrams = csr_matrix(np.array(bcbigrams_list))
	X = hstack([X_unigrams,X_bigrams,X_bcunigrams,X_bcbigrams])
	X = csr_matrix(X)
	words_arc_feature_len = len(words_arc_dict.keys())

	for one_list in X_list_word_arc:
		zero_list = [0]*words_arc_feature_len
		for j in one_list:
			zero_list[j] = 1
		X_word.append(zero_list)

	BC_arc_feature_len = len(BC_arc_dict.keys())

	for one_list in X_list_BC_arc:
		zero_list = [0]*BC_arc_feature_len
		for j in one_list:
			zero_list[j] = 1
		X_BC.append(zero_list)

	X_word = csr_matrix(np.array(X_word))
	X_BC = csr_matrix(np.array(X_BC))
	X = hstack([X, X_BC, X_word])
	del X_word, X_BC
	X_POS = csr_matrix(np.array(Part_of_speech_list))
	X_Pron = csr_matrix(np.array(Pronunciation_list))
	X_Cap = csr_matrix(np.array(Cap_list))
	X_whole_senti = csr_matrix(np.array(Tweet_whole_sentiment_list))
	X_word_senti = csr_matrix(np.array(Tweet_word_sentiment_list))
	X_intensifier = csr_matrix(np.array(intensifier_list))
	X = hstack([X, X_POS, X_Pron, X_Cap, X_whole_senti, X_word_senti, X_intensifier])
	X = csr_matrix(X)
	print X.shape
	Y_target = np.array(target_list)
	del target_list
	result = LR_model(name_dict, X, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])

# Generate the perdicting accuracy using all Features
def baseline():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	# author features
	all_hist_list = []
	tfidf_dict = {}
	fit_tfidf_list = []
	profile_nofer_list = []
	profile_nofing_list = []
	profile_not_list = []
	profile_duration_list = []
	profile_avgt_list = []
	profile_verif_list = []
	profile_tz_list = []
	sentiment_list = []
	unigrams_list = []

	# tweet features
	tweet_features_unigrams_list = []
	tweet_features_bigrams_list = []

	bigram_dict = {}
	bcunigrams_list = []
	bcbigrams_list = []

	words_arc_dict = {}
	index = 0
	BC_arc_dict = {}
	BC_index = 0
	X_list_word_arc = []
	X_list_BC_arc = []
	X_word = []
	X_BC = []

	TagList = ['N','O','S','^','Z','L','M','V','A','R',
	   			'!','D','P','&','T','X','Y','#','@','~',
				'U','E','$',',','G']
	Part_of_speech_list = []
	Pronunciation_list = []
	Cap_list = []
	senti_dict ={key:value for key, value in zip(["very negative", "negative", "neutral", "positive", "very positive"], range(5))}
	original_whole_sentiment = [0, 0, 0, 0, 0]
	Tweet_whole_sentiment_list = []
	Tweet_word_sentiment_list = []
	intensifier_list = []

	# audience features
	audience_features_X = []
	audience_features_tfidf_dict = {}
	audience_features_fit_tfidf_list = []
	audience_features_unigrams_list = []
	audience_features_author_hist_list = []
	audience_features_audience_hist_list = []
	audience_features_profile_nofer_list = []
	audience_features_profile_nofing_list = []
	audience_features_profile_duration_list = []
	audience_features_profile_avgt_list = []
	audience_features_profile_veri_yes_list = []
	audience_features_profile_veri_no_list = []
	audience_features_profile_not_list = []

	# environment features
	X_environment_features = np.array([])
	unigrams_list_environment_features = []
	BC_values = BC_original_tweet.get_cluster_dic()[1]
	temp_keys = BC_values.keys()
	length_environment_features = len(temp_keys)

	for i in range(length_environment_features):
		BC_values[temp_keys[i]] = i


	target_list = []
	name_dict = {}

	for i in xrange(dbtweets.find({'all_features':{'$exists':True}}).count()): 
	#for i in xrange(2000):
		document = dbtweets.find({'all_features':{'$exists':True}})[i]
		BC = document['BrownCluster']
		for j in range(len(BC)-1):
			bi_BC = (BC[j],BC[j+1])
			if bi_BC not in bigram_dict:
				bigram_dict[bi_BC] = 1
	bigram_dict_keys = bigram_dict.keys()
	for i in xrange(dbtweets.find({'all_features':{'$exists':True}}).count()): 
	#for i in xrange(2000):
		try:
			document = dbtweets.find({'all_features':{'$exists':True}})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			hist_list = document['hist_list']
			profile_nofer = document['followers_count']
			profile_nofing = document['following_count']
			profile_not = document['tweets_count']
			profile_duration = float(document['duration'])
			profile_avgt = document['avg_tweet']
			profile_verif = document['verified'].encode('utf-8')
			profile_tz = document['time_zone'].encode('utf-8')
			positive = float(document['hist_sentiment_positive'])
			very_positive = float(document['hist_sentiment_very_positive'])
			negative = float(document['hist_sentiment_negative'])
			very_negative = float(document['hist_sentiment_very_negative'])
			neutral = float(document['hist_sentiment_neutral'])
			profile_unigrams = document['profile'].encode('utf-8')
			word_unigrams = document['tweet_text'].encode('utf-8')
			BC = document['BrownCluster']
			original_tweet = document['original_tweet'].encode('utf-8')
			original_BC = document['original_tweet_BrownCluster']
			audience_full_name = document['audience']
			audience_hist = document['audience_hist_list']
			audience_features_profile_nofer = document['audience_followers_count']
			audience_features_profile_nofing = document['audience_following_count']
			audience_features_profile_duration = document['audience_duration']
			audience_features_profile_avgt = document['audience_avg_tweet']
			audience_features_profile_veri_yes = document['audience_verified_yes']
			audience_features_profile_veri_no = document['audience_verified_no']
			audience_features_profile_not = audience_features_profile_avgt * audience_features_profile_duration
			audience_features_profile = document['audience_profile'].encode('utf-8')

			audience_features_unigrams_list.append(audience_features_profile)
			audience_features_profile_nofer_list.append([audience_features_profile_nofer])
			audience_features_profile_nofing_list.append([audience_features_profile_nofing])
			audience_features_profile_not_list.append([audience_features_profile_not])
			audience_features_profile_duration_list.append([audience_features_profile_duration])
			audience_features_profile_avgt_list.append([audience_features_profile_avgt])
			audience_features_profile_veri_yes_list.append([audience_features_profile_veri_yes])
			audience_features_profile_veri_no_list.append([audience_features_profile_veri_no])

			author_at_name_dict = find_at_names_in_hist_tweets(hist_list)
			audience_at_name_dict = find_at_names_in_hist_tweets(audience_hist)

			message_sent = 0

			if author_full_name in audience_at_name_dict:
				message_sent = audience_at_name_dict[author_full_name]

			rank = 0.0

			if message_sent != 0:
				v = audience_at_name_dict.values()
				index = sorted(v).index(message_sent)
				rank = 	float(index + 1)/len(v) 

			mutual_meassage = 0

			if author_full_name in audience_at_name_dict:
				for one_aud in audience_full_name:
					if one_aud in author_at_name_dict:
						mutual_meassage = 1
						break

			audience_features_X.append([message_sent, rank, mutual_meassage])


			len_of_hist_list = 0
			rand_order = random.sample(range(1, len(hist_list)), len(hist_list)-1)
			modified_hist_list = ''

			while(len_of_hist_list<1000):
				try:
					tweet_add = hist_list[rand_order.pop()].encode('utf-8')
					len_of_hist_list += len(tweet_add)
					modified_hist_list = modified_hist_list +  tweet_add + '\n' 
				except IndexError:
					break
			
			audience_features_author_hist_list.append(modified_hist_list)

			len_of_hist_list = 0
			rand_order = random.sample(range(1, len(audience_hist)), len(audience_hist)-1)
			modified_hist_list = ''

			while(len_of_hist_list<1000):
				try:
					tweet_add = audience_hist[rand_order.pop()].encode('utf-8')
					len_of_hist_list += len(tweet_add)
					modified_hist_list = modified_hist_list +  tweet_add + '\n' 
				except IndexError:
					break
			
			audience_features_audience_hist_list.append(modified_hist_list)

			top_tfidf_list = []
			vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,1), stop_words='english')
			tfidf_matrix = vectorizer.fit_transform(audience_hist)
			idf = vectorizer.idf_
			scores =  dict(zip(vectorizer.get_feature_names(), idf))
			sortedList = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
			top_100_number = len(sortedList)
			if top_100_number < 100:
				for n in range(top_100_number):
					top_tfidf_list.append(sortedList[n][0])
			else:
				for n in range(100):
					top_tfidf_list.append(sortedList[n][0])
			
			for term in top_tfidf_list:
				if term not in audience_features_tfidf_dict:
					audience_features_tfidf_dict[term] = 1
				else:
					continue
			input_data = " ".join(top_tfidf_list)
			audience_features_fit_tfidf_list.append(input_data)

			# environment features
			unigrams_list_environment_features.append(original_tweet)

			pairwise_BC = [0]*length_environment_features
			for bc in BC:
				temp_index = BC_values[bc]
				pairwise_BC[temp_index] = 1

			pairwise_OBC = [0]*length_environment_features
			for obc in original_BC:
				temp_index = BC_values[obc]
				pairwise_OBC[temp_index] = 1

			pairwise_BC = np.array(pairwise_BC)
			pairwise_OBC = np.array(pairwise_OBC)
			#pairwise = pairwise_BC * pairwise_OBC
			pairwise = pairwise_BC | pairwise_OBC

			if X_environment_features.shape[0]==0:
				X_environment_features = pairwise
			else:
				X_environment_features = np.vstack((X_environment_features, pairwise))

			# Tweet feature part
			Pronunciation_list.append([document["number_Polysyllables "],document["number_no_vowels"]])
			tweet_features_unigrams_list.append(word_unigrams)
			tweet_features_bigrams_list.append(word_unigrams)
			cluster1000_d = BrownCluster.get_cluster_dic()[1]
			cluster1000 = cluster1000_d.keys()
			bcword_unigrams = [0]*len(cluster1000)
			bcword_bigrams = [0]*len(bigram_dict_keys)
			for j in BC:
				bcword_unigrams[cluster1000.index(j)] = 1
			for j in range(len(BC)-1):
				bi_BC = (BC[j],BC[j+1])
				bcword_bigrams[bigram_dict_keys.index(bi_BC)] = 1

			bcunigrams_list.append(bcword_unigrams)
			bcbigrams_list.append(bcword_bigrams)
			two_words_arc_list = []
			two_BC_arc_list = []
			two_words_arc = document['two_words_arc']
			two_BC_arc = document['two_BC_arc']

			for one_arc in two_words_arc:
				if one_arc not in words_arc_dict:
					words_arc_dict[one_arc] = index
					index += 1
				two_words_arc_list.append(words_arc_dict[one_arc])
			X_list_word_arc.append(two_words_arc_list)

			for one_arc in two_BC_arc:
				if one_arc not in BC_arc_dict:
					BC_arc_dict[one_arc] = BC_index
					BC_index += 1
				two_BC_arc_list.append(BC_arc_dict[one_arc])
			X_list_BC_arc.append(two_BC_arc_list)

			row_feature = []
			for a in TagList:			
				a_ratio = a+'_ratio'
				if a== '$':
					a = 'numeral_pos'
					a_ratio = a+'_ratio'
				a_value = document[a]
				a_ratio_value = document[a_ratio]
				row_feature.append(a_value)
				row_feature.append(a_ratio_value)
			row_feature.append(document['lexical_density'])
			Part_of_speech_list.append(row_feature)

			cap_row_feature = []
			for a in TagList:			
				a_cap_count = a+'_cap_count'
				if a== '$':
					a_cap_count = 'numeral_pos_cap'
				a_value = document[a_cap_count]
				cap_row_feature.append(a_value)

			cap_row_feature.append(document['ini_cap_number'])
			cap_row_feature.append(document['all_cap_number'])

			Cap_list.append(cap_row_feature)
			this_sentence_sentiment = []
			this_sentence_sentiment.append(float(document["positivenode"]))
			this_sentence_sentiment.append(float(document["negativenode"]))
			whole_sentiment = document['tweet_whole_sentimentpredict']
			original_whole_sentiment[senti_dict[whole_sentiment]] = 1
			this_sentence_sentiment = this_sentence_sentiment + original_whole_sentiment
			Tweet_whole_sentiment_list.append(this_sentence_sentiment)
			Tweet_word_sentiment_list.append([document["effect_distance"],document["min_word_effect"],document["max_word_effect"],document["sentiment_distance"],document["min_word_senti"],document["max_word_senti"]])
			intensifier = document['intensifier']
			intensifier_list.append([intensifier])

			# end

			# author feature 
			len_of_hist_list = 0
			rand_order = random.sample(range(1, len(hist_list)), len(hist_list)-1)
			modified_hist_list = ''

			while(len_of_hist_list<1000):
				try:
					tweet_add = hist_list[rand_order.pop()].encode('utf-8')
					len_of_hist_list += len(tweet_add)
					modified_hist_list = modified_hist_list + '\n' + tweet_add
				except IndexError:
					break
			
			all_hist_list.append(modified_hist_list)

			top_tfidf_list = []
			vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,1), stop_words='english')
			tfidf_matrix = vectorizer.fit_transform(hist_list)
			idf = vectorizer.idf_
			scores =  dict(zip(vectorizer.get_feature_names(), idf))
			sortedList = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
			top_100_number = len(sortedList)
			if top_100_number < 100:
				for n in range(top_100_number):
					top_tfidf_list.append(sortedList[n][0])
			else:
				for n in range(100):
					top_tfidf_list.append(sortedList[n][0])
			
			for term in top_tfidf_list:
				if term not in tfidf_dict:
					tfidf_dict[term] = 1
				else:
					continue
			input_data = " ".join(top_tfidf_list)
			fit_tfidf_list.append(input_data)

			profile_nofer_list.append([profile_nofer])
			profile_nofing_list.append([profile_nofing])
			profile_not_list.append([profile_not])
			profile_duration_list.append([profile_duration])
			profile_avgt_list.append([profile_avgt])
			profile_verif_list.append(profile_verif)
			profile_tz_list.append(profile_tz)
			sentiment_list.append([positive, very_positive, negative, very_negative, neutral])
			
			unigrams_list.append(profile_unigrams)
			# end

			sarcasm_score = document['sarcasm_score']			
			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(all_hist_list)-1)
			else:
				name_dict[author_full_name] = [len(all_hist_list)-1]
		except ValueError:
			continue
	print '0'
	client.close()

	# author feautre
	X = BOW.Get_unigrams(all_hist_list)[1]
	count_vect = CountVectorizer(vocabulary=tfidf_dict.keys(),binary=True)
	del tfidf_dict, all_hist_list
	X_salient_terms = count_vect.fit_transform(fit_tfidf_list)
	del fit_tfidf_list, count_vect
	profile_nofer_list = preprocess_num(np.array(profile_nofer_list))
	profile_nofing_list = preprocess_num(np.array(profile_nofing_list))
	profile_not_list = preprocess_num(np.array(profile_not_list))
	profile_duration_list = preprocess_num(np.array(profile_duration_list))
	profile_avgt_list = preprocess_num(np.array(profile_avgt_list))
	
	count_vect = CountVectorizer()
	profile_verif_list = (count_vect.fit_transform(profile_verif_list)).toarray()
	count_vect = CountVectorizer()
	profile_tz_list = (count_vect.fit_transform(profile_tz_list)).toarray()
	
	X_profile_info = np.concatenate((profile_nofer_list, profile_nofing_list, profile_not_list, 
						profile_duration_list, profile_avgt_list, profile_verif_list, 
						profile_tz_list), axis=1)
	del profile_nofer_list, profile_nofing_list, profile_not_list, profile_duration_list, profile_avgt_list, profile_verif_list, profile_tz_list
	X_profile_info = csr_matrix(X_profile_info)
	
	X_sentiment = csr_matrix(np.array(sentiment_list))
	count_vect = CountVectorizer(stop_words='english',binary=True)
	unigrams_list = count_vect.fit_transform(unigrams_list)
	del count_vect
	X_other = hstack([X_profile_info, X_salient_terms, X_sentiment, unigrams_list])
	X_other = csr_matrix(X_other)
	del X_profile_info, X_salient_terms, X_sentiment, unigrams_list
	# end

	print '1'
	# tweet feature 
	unicount_vect = CountVectorizer(ngram_range=(1,1), lowercase = True,  stop_words='english', min_df=1)
	X_unigrams = unicount_vect.fit_transform(tweet_features_unigrams_list)
	bicount_vect = CountVectorizer(ngram_range=(2,2), lowercase = True, stop_words='english',  min_df=1)
	X_bigrams = bicount_vect.fit_transform(tweet_features_bigrams_list)

	X_bcunigrams = csr_matrix(np.array(bcunigrams_list))
	X_bcbigrams = csr_matrix(np.array(bcbigrams_list))
	X_author_features = hstack([X_unigrams,X_bigrams,X_bcunigrams,X_bcbigrams])
	del X_unigrams,X_bigrams,X_bcunigrams,X_bcbigrams
	X_author_features = csr_matrix(X_author_features)
	words_arc_feature_len = len(words_arc_dict.keys())
	print '2'
	for one_list in X_list_word_arc:
		zero_list = [0]*words_arc_feature_len
		for j in one_list:
			zero_list[j] = 1
		X_word.append(zero_list)

	BC_arc_feature_len = len(BC_arc_dict.keys())

	for one_list in X_list_BC_arc:
		zero_list = [0]*BC_arc_feature_len
		for j in one_list:
			zero_list[j] = 1
		X_BC.append(zero_list)

	X_word = csr_matrix(np.array(X_word))
	X_BC = csr_matrix(np.array(X_BC))
	X_author_features = hstack([X_author_features, X_BC, X_word])
	print '3'
	del X_word, X_BC
	X_POS = csr_matrix(np.array(Part_of_speech_list))
	X_Pron = csr_matrix(np.array(Pronunciation_list))
	X_Cap = csr_matrix(np.array(Cap_list))
	X_whole_senti = csr_matrix(np.array(Tweet_whole_sentiment_list))
	X_word_senti = csr_matrix(np.array(Tweet_word_sentiment_list))
	X_intensifier = csr_matrix(np.array(intensifier_list))
	X_author_features = hstack([X_author_features, X_POS, X_Pron, X_Cap, X_whole_senti, X_word_senti, X_intensifier])
	X_author_features = csr_matrix(X_author_features)
	del X_POS, X_Pron, X_Cap, X_whole_senti, X_word_senti, X_intensifier
	# end

	X_other = hstack([X_author_features, X_other])
	del X_author_features
	X_other = csr_matrix(X_other)
	print '4'

	# environemnt features
	vectorizer = CountVectorizer(min_df=1, stop_words='english',binary=True)
	X_uni_environment_features = vectorizer.fit_transform(unigrams_list_environment_features)
	X_environment_features = csr_matrix(X_environment_features)
	X_environment_features = hstack([X_environment_features, X_uni_environment_features])

	X_other = hstack([X_environment_features, X_other])
	del X_environment_features, X_uni_environment_features 
	X_other = csr_matrix(X_other)

	# audience features
	X_aud_hist_topics = BOW.Get_unigrams(audience_features_audience_hist_list)[1]
	X_all = [a+b for a,b in zip(audience_features_author_hist_list, audience_features_audience_hist_list)]
	unicount_vect = CountVectorizer(ngram_range=(1,1), lowercase = True, stop_words='english', min_df=1)
	unicount_vect.fit(X_all)
	audience_features_X_author = unicount_vect.transform(audience_features_author_hist_list)
	audience_features_X_audience = unicount_vect.transform(audience_features_audience_hist_list)

	vectorizer = CountVectorizer(min_df=1, stop_words='english')
	audience_features_X_unigrams = vectorizer.fit_transform(audience_features_unigrams_list)

	count_vect = CountVectorizer(vocabulary=audience_features_tfidf_dict.keys(),binary=True)
	audience_features_X_tfidf = count_vect.fit_transform(audience_features_fit_tfidf_list)


	print '5'
	audience_features_profile_nofer_list = preprocess_num(np.array(audience_features_profile_nofer_list))
	audience_features_profile_nofing_list = preprocess_num(np.array(audience_features_profile_nofing_list))
	audience_features_profile_not_list = preprocess_num(np.array(audience_features_profile_not_list))
	audience_features_profile_duration_list = preprocess_num(np.array(audience_features_profile_duration_list))
	audience_features_profile_avgt_list = preprocess_num(np.array(audience_features_profile_avgt_list))
	audience_features_profile_veri_yes_list = np.array(audience_features_profile_veri_yes_list)
	audience_features_profile_veri_no_list = np.array(audience_features_profile_veri_no_list)

	audience_features_X_profile_info = np.concatenate((audience_features_profile_nofer_list, audience_features_profile_nofing_list, audience_features_profile_duration_list,
						audience_features_profile_avgt_list, audience_features_profile_veri_yes_list, audience_features_profile_veri_no_list,
						audience_features_profile_not_list), axis=1)
	del audience_features_profile_nofer_list, audience_features_profile_nofing_list, audience_features_profile_duration_list,audience_features_profile_avgt_list, audience_features_profile_veri_yes_list, audience_features_profile_veri_no_list,audience_features_profile_not_list
	audience_features_X = np.hstack((np.array(audience_features_X),audience_features_X_profile_info))
	del audience_features_X_profile_info
	audience_features_X_his_com = csr_matrix(audience_features_X)
	audience_features_X_his_com = hstack([audience_features_X_his_com, audience_features_X_tfidf, audience_features_X_unigrams])
	del audience_features_X_tfidf, audience_features_X_unigrams
	audience_features_X_his_com = csr_matrix(audience_features_X_his_com)

	X_other = hstack([audience_features_X_his_com, X_other])
	del audience_features_X_his_com
	X_other = csr_matrix(X_other)
	print '6'
	Y = np.array(target_list)
	del target_list
	name_list = name_dict.keys()
	total_set_length = len(name_list)
	kf = KFold(total_set_length, n_folds=5)
	avg_score = []
	avg_auc_score = []
	for train_index, test_index in kf:
		temp_index = []
		for i in train_index:
			temp_index += name_dict[name_list[i]]
		train_index = np.array(temp_index)
		print '7'
		temp_index = []
		for j in test_index:
			temp_index += name_dict[name_list[j]]
		test_index = np.array(temp_index)
		print '8'
		X_train, X_test = X[train_index], X[test_index]
		######
		model = lda.LDA(n_topics=100, n_iter=300, random_state=1)
		X_train = csr_matrix(X_train)
		model.fit(X_train)
		X_train = model.doc_topic_
		X_test = model.transform(X_test)
		X_test = csr_matrix(X_test)
		
		X_aud_hist_topics_train, X_aud_hist_topics_test = X_aud_hist_topics[train_index], X_aud_hist_topics[test_index]
		model = lda.LDA(n_topics=100, n_iter=300, random_state=1)
		X_aud_hist_topics_train = csr_matrix(X_aud_hist_topics_train)
		model.fit(X_aud_hist_topics_train)
		X_aud_hist_topics_train = model.doc_topic_
		X_aud_hist_topics_train = csr_matrix(X_aud_hist_topics_train)
		X_aud_hist_topics_test = model.transform(X_aud_hist_topics_test)
		X_aud_hist_topics_test = csr_matrix(X_aud_hist_topics_test)

		X_train = hstack([X_train,X_aud_hist_topics_train])
		X_test = hstack([X_test, X_aud_hist_topics_test])
		del X_aud_hist_topics_test, X_aud_hist_topics_train
		######


		print '9'
		X_author_train, X_author_test = audience_features_X_author[train_index], audience_features_X_author[test_index]
		X_audience_train, X_audience_test = audience_features_X_audience[train_index], audience_features_X_audience[test_index]
		
		X_author_train_index = X_author_train.shape[0]
		audience_features_X_train = vstack([X_author_train, X_audience_train])
		audience_features_X_train = csr_matrix(audience_features_X_train)
		model = lda.LDA(n_topics=100, n_iter=300, random_state=1)
		model.fit(audience_features_X_train)
		del audience_features_X_train
		doc_topic = model.doc_topic_
		X_author_train = doc_topic[:X_author_train_index]
		X_audience_train = doc_topic[X_author_train_index:]
		del doc_topic
		print '10'
		audience_features_X_train = abs(X_author_train - X_audience_train)
		del X_author_train, X_audience_train
		min_max_scaler = preprocessing.MinMaxScaler()
		audience_features_X_train = min_max_scaler.fit_transform(audience_features_X_train)
		audience_features_X_train = csr_matrix(audience_features_X_train)
		X_train = hstack([X_train,audience_features_X_train])
		del audience_features_X_train

		print '11'
		X_author_test = X_author_test.toarray()
		X_audience_test = X_audience_test.toarray()
		X_author_test = model.transform(X_author_test)
		X_audience_test = model.transform(X_audience_test)
		audience_features_X_test = abs(X_author_test - X_audience_test)
		del X_author_test,X_audience_test
		min_max_scaler = preprocessing.MinMaxScaler()
		audience_features_X_test = min_max_scaler.fit_transform(audience_features_X_test)
		audience_features_X_test = csr_matrix(audience_features_X_test)
		X_test = hstack([X_test,audience_features_X_test])
		del audience_features_X_test
		print '12'
		#


		X_other_train, X_other_test = X_other[train_index], X_other[test_index]
		X_train = hstack([X_train, X_other_train])
		X_test = hstack([X_test, X_other_test])
		del X_other_train, X_other_test
		print '13'
		Y_train, Y_test = Y[train_index], Y[test_index]

		parameters = {'tol':[0.001,0.0001], 'C':[0.00001, 0.0001, 0.001, 0.1, 1, 10]}
		lr = linear_model.LogisticRegression(penalty='l2')
		clf = grid_search.GridSearchCV(lr, parameters)
		clf.fit(X_train, Y_train)
		y_true, y_pred = Y_test, clf.predict(X_test)
		score = clf.score(X_test, Y_test)
		avg_score.append(score)
		auc = roc_auc_score(y_true, y_pred)
		avg_auc_score.append(auc)
		print score, auc

	accuracy = reduce(lambda x, y: x + y, avg_score) / len(avg_score)
	auc_score = reduce(lambda x, y: x + y, avg_auc_score) / len(avg_auc_score)
	print 'accuracy is: %s, auc score is:%s'%(accuracy, auc_score)

# Tweet + Environemnt features
# Generate the perdicting accuracy using all Features
def Tweet_Environemnt_features():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets
	# tweet features
	tweet_features_unigrams_list = []
	tweet_features_bigrams_list = []

	bigram_dict = {}
	bcunigrams_list = []
	bcbigrams_list = []

	words_arc_dict = {}
	index = 0
	BC_arc_dict = {}
	BC_index = 0
	X_list_word_arc = []
	X_list_BC_arc = []
	X_word = []
	X_BC = []

	TagList = ['N','O','S','^','Z','L','M','V','A','R',
	   			'!','D','P','&','T','X','Y','#','@','~',
				'U','E','$',',','G']
	Part_of_speech_list = []
	Pronunciation_list = []
	Cap_list = []
	senti_dict ={key:value for key, value in zip(["very negative", "negative", "neutral", "positive", "very positive"], range(5))}
	original_whole_sentiment = [0, 0, 0, 0, 0]
	Tweet_whole_sentiment_list = []
	Tweet_word_sentiment_list = []
	intensifier_list = []

	# environment features
	X_environment_features = np.array([])
	unigrams_list_environment_features = []
	BC_values = BC_original_tweet.get_cluster_dic()[1]
	temp_keys = BC_values.keys()
	length_environment_features = len(temp_keys)

	for i in range(length_environment_features):
		BC_values[temp_keys[i]] = i


	target_list = []
	name_dict = {}

	for i in xrange(dbtweets.find({'all_features':{'$exists':True}}).count()): 
	#for i in xrange(500):
		document = dbtweets.find({'all_features':{'$exists':True}})[i]
		BC = document['BrownCluster']
		for j in range(len(BC)-1):
			bi_BC = (BC[j],BC[j+1])
			if bi_BC not in bigram_dict:
				bigram_dict[bi_BC] = 1
	bigram_dict_keys = bigram_dict.keys()
	for i in xrange(dbtweets.find({'all_features':{'$exists':True}}).count()): 
	#for i in xrange(500):
		try:
			document = dbtweets.find({'all_features':{'$exists':True}})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			hist_list = document['hist_list']
			profile_nofer = document['followers_count']
			profile_nofing = document['following_count']
			profile_not = document['tweets_count']
			profile_duration = float(document['duration'])
			profile_avgt = document['avg_tweet']
			profile_verif = document['verified'].encode('utf-8')
			profile_tz = document['time_zone'].encode('utf-8')
			positive = float(document['hist_sentiment_positive'])
			very_positive = float(document['hist_sentiment_very_positive'])
			negative = float(document['hist_sentiment_negative'])
			very_negative = float(document['hist_sentiment_very_negative'])
			neutral = float(document['hist_sentiment_neutral'])
			profile_unigrams = document['profile'].encode('utf-8')
			word_unigrams = document['tweet_text'].encode('utf-8')
			BC = document['BrownCluster']
			original_tweet = document['original_tweet'].encode('utf-8')
			original_BC = document['original_tweet_BrownCluster']

			# environment features
			unigrams_list_environment_features.append(original_tweet)

			pairwise_BC = [0]*length_environment_features
			for bc in BC:
				temp_index = BC_values[bc]
				pairwise_BC[temp_index] = 1

			pairwise_OBC = [0]*length_environment_features
			for obc in original_BC:
				temp_index = BC_values[obc]
				pairwise_OBC[temp_index] = 1

			pairwise_BC = np.array(pairwise_BC)
			pairwise_OBC = np.array(pairwise_OBC)
			#pairwise = pairwise_BC * pairwise_OBC
			pairwise = pairwise_BC | pairwise_OBC

			if X_environment_features.shape[0]==0:
				X_environment_features = pairwise
			else:
				X_environment_features = np.vstack((X_environment_features, pairwise))

			# Tweet feature part
			Pronunciation_list.append([document["number_Polysyllables "],document["number_no_vowels"]])
			tweet_features_unigrams_list.append(word_unigrams)
			tweet_features_bigrams_list.append(word_unigrams)
			cluster1000_d = BrownCluster.get_cluster_dic()[1]
			cluster1000 = cluster1000_d.keys()
			bcword_unigrams = [0]*len(cluster1000)
			bcword_bigrams = [0]*len(bigram_dict_keys)
			for j in BC:
				bcword_unigrams[cluster1000.index(j)] = 1
			for j in range(len(BC)-1):
				bi_BC = (BC[j],BC[j+1])
				bcword_bigrams[bigram_dict_keys.index(bi_BC)] = 1

			bcunigrams_list.append(bcword_unigrams)
			bcbigrams_list.append(bcword_bigrams)
			two_words_arc_list = []
			two_BC_arc_list = []
			two_words_arc = document['two_words_arc']
			two_BC_arc = document['two_BC_arc']

			for one_arc in two_words_arc:
				if one_arc not in words_arc_dict:
					words_arc_dict[one_arc] = index
					index += 1
				two_words_arc_list.append(words_arc_dict[one_arc])
			X_list_word_arc.append(two_words_arc_list)

			for one_arc in two_BC_arc:
				if one_arc not in BC_arc_dict:
					BC_arc_dict[one_arc] = BC_index
					BC_index += 1
				two_BC_arc_list.append(BC_arc_dict[one_arc])
			X_list_BC_arc.append(two_BC_arc_list)

			row_feature = []
			for a in TagList:			
				a_ratio = a+'_ratio'
				if a== '$':
					a = 'numeral_pos'
					a_ratio = a+'_ratio'
				a_value = document[a]
				a_ratio_value = document[a_ratio]
				row_feature.append(a_value)
				row_feature.append(a_ratio_value)
			row_feature.append(document['lexical_density'])
			Part_of_speech_list.append(row_feature)

			cap_row_feature = []
			for a in TagList:			
				a_cap_count = a+'_cap_count'
				if a== '$':
					a_cap_count = 'numeral_pos_cap'
				a_value = document[a_cap_count]
				cap_row_feature.append(a_value)

			cap_row_feature.append(document['ini_cap_number'])
			cap_row_feature.append(document['all_cap_number'])

			Cap_list.append(cap_row_feature)
			this_sentence_sentiment = []
			this_sentence_sentiment.append(float(document["positivenode"]))
			this_sentence_sentiment.append(float(document["negativenode"]))
			whole_sentiment = document['tweet_whole_sentimentpredict']
			original_whole_sentiment[senti_dict[whole_sentiment]] = 1
			this_sentence_sentiment = this_sentence_sentiment + original_whole_sentiment
			Tweet_whole_sentiment_list.append(this_sentence_sentiment)
			Tweet_word_sentiment_list.append([document["effect_distance"],document["min_word_effect"],document["max_word_effect"],document["sentiment_distance"],document["min_word_senti"],document["max_word_senti"]])
			intensifier = document['intensifier']
			intensifier_list.append([intensifier])

			# end
			sarcasm_score = document['sarcasm_score']			
			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(target_list)-1)
			else:
				name_dict[author_full_name] = [len(target_list)-1]
		except ValueError:
			continue
	client.close()
	# tweet feature 
	unicount_vect = CountVectorizer(ngram_range=(1,1), lowercase = True,  stop_words='english', min_df=1)
	X_unigrams = unicount_vect.fit_transform(tweet_features_unigrams_list)
	bicount_vect = CountVectorizer(ngram_range=(2,2), lowercase = True, stop_words='english',  min_df=1)
	X_bigrams = bicount_vect.fit_transform(tweet_features_bigrams_list)

	X_bcunigrams = csr_matrix(np.array(bcunigrams_list))
	X_bcbigrams = csr_matrix(np.array(bcbigrams_list))
	X_author_features = hstack([X_unigrams,X_bigrams,X_bcunigrams,X_bcbigrams])
	del X_unigrams,X_bigrams,X_bcunigrams,X_bcbigrams
	X_author_features = csr_matrix(X_author_features)
	words_arc_feature_len = len(words_arc_dict.keys())
	for one_list in X_list_word_arc:
		zero_list = [0]*words_arc_feature_len
		for j in one_list:
			zero_list[j] = 1
		X_word.append(zero_list)

	BC_arc_feature_len = len(BC_arc_dict.keys())

	for one_list in X_list_BC_arc:
		zero_list = [0]*BC_arc_feature_len
		for j in one_list:
			zero_list[j] = 1
		X_BC.append(zero_list)

	X_word = csr_matrix(np.array(X_word))
	X_BC = csr_matrix(np.array(X_BC))
	X_author_features = hstack([X_author_features, X_BC, X_word])
	del X_word, X_BC
	X_POS = csr_matrix(np.array(Part_of_speech_list))
	X_Pron = csr_matrix(np.array(Pronunciation_list))
	X_Cap = csr_matrix(np.array(Cap_list))
	X_whole_senti = csr_matrix(np.array(Tweet_whole_sentiment_list))
	X_word_senti = csr_matrix(np.array(Tweet_word_sentiment_list))
	X_intensifier = csr_matrix(np.array(intensifier_list))
	X_author_features = hstack([X_author_features, X_POS, X_Pron, X_Cap, X_whole_senti, X_word_senti, X_intensifier])
	X_author_features = csr_matrix(X_author_features)
	del X_POS, X_Pron, X_Cap, X_whole_senti, X_word_senti, X_intensifier
	X_other = X_author_features
	del X_author_features
	X_other = csr_matrix(X_other)
	# environemnt features
	vectorizer = CountVectorizer(min_df=1, stop_words='english',binary=True)
	X_uni_environment_features = vectorizer.fit_transform(unigrams_list_environment_features)
	X_environment_features = csr_matrix(X_environment_features)
	X_environment_features = hstack([X_environment_features, X_uni_environment_features])

	X_other = hstack([X_environment_features, X_other])
	del X_environment_features, X_uni_environment_features 
	X_other = csr_matrix(X_other)
	Y_target = np.array(target_list)
	del target_list
	result = LR_model(name_dict, X_other, Y_target)
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])


# Generate the perdicting accuracy Tweet Features + Audience Features
def Tweet_Audience_Features():
	# Connect to MongoDB
	client = MongoClient('127.0.0.1', 27017)
	db = client['IronyHQ']
	dbtweets = db.tweets

	# tweet features
	tweet_features_unigrams_list = []
	tweet_features_bigrams_list = []

	bigram_dict = {}
	bcunigrams_list = []
	bcbigrams_list = []

	words_arc_dict = {}
	index = 0
	BC_arc_dict = {}
	BC_index = 0
	X_list_word_arc = []
	X_list_BC_arc = []
	X_word = []
	X_BC = []

	TagList = ['N','O','S','^','Z','L','M','V','A','R',
	   			'!','D','P','&','T','X','Y','#','@','~',
				'U','E','$',',','G']
	Part_of_speech_list = []
	Pronunciation_list = []
	Cap_list = []
	senti_dict ={key:value for key, value in zip(["very negative", "negative", "neutral", "positive", "very positive"], range(5))}
	original_whole_sentiment = [0, 0, 0, 0, 0]
	Tweet_whole_sentiment_list = []
	Tweet_word_sentiment_list = []
	intensifier_list = []

	# audience features
	audience_features_X = []
	audience_features_tfidf_dict = {}
	audience_features_fit_tfidf_list = []
	audience_features_unigrams_list = []
	audience_features_author_hist_list = []
	audience_features_audience_hist_list = []
	audience_features_profile_nofer_list = []
	audience_features_profile_nofing_list = []
	audience_features_profile_duration_list = []
	audience_features_profile_avgt_list = []
	audience_features_profile_veri_yes_list = []
	audience_features_profile_veri_no_list = []
	audience_features_profile_not_list = []

	target_list = []
	name_dict = {}

	for i in xrange(dbtweets.find({'all_features':{'$exists':True}}).count()): 
	#for i in xrange(500):
		document = dbtweets.find({'all_features':{'$exists':True}})[i]
		BC = document['BrownCluster']
		for j in range(len(BC)-1):
			bi_BC = (BC[j],BC[j+1])
			if bi_BC not in bigram_dict:
				bigram_dict[bi_BC] = 1
	bigram_dict_keys = bigram_dict.keys()
	for i in xrange(dbtweets.find({'all_features':{'$exists':True}}).count()): 
	#for i in xrange(500):
		try:
			document = dbtweets.find({'all_features':{'$exists':True}})[i]
			author_full_name = document['author_full_name'].encode('utf-8')
			hist_list = document['hist_list']
			profile_unigrams = document['profile'].encode('utf-8')
			word_unigrams = document['tweet_text'].encode('utf-8')
			BC = document['BrownCluster']
			original_tweet = document['original_tweet'].encode('utf-8')
			original_BC = document['original_tweet_BrownCluster']
			audience_full_name = document['audience']
			audience_hist = document['audience_hist_list']
			audience_features_profile_nofer = document['audience_followers_count']
			audience_features_profile_nofing = document['audience_following_count']
			audience_features_profile_duration = document['audience_duration']
			audience_features_profile_avgt = document['audience_avg_tweet']
			audience_features_profile_veri_yes = document['audience_verified_yes']
			audience_features_profile_veri_no = document['audience_verified_no']
			audience_features_profile_not = audience_features_profile_avgt * audience_features_profile_duration
			audience_features_profile = document['audience_profile'].encode('utf-8')

			audience_features_unigrams_list.append(audience_features_profile)
			audience_features_profile_nofer_list.append([audience_features_profile_nofer])
			audience_features_profile_nofing_list.append([audience_features_profile_nofing])
			audience_features_profile_not_list.append([audience_features_profile_not])
			audience_features_profile_duration_list.append([audience_features_profile_duration])
			audience_features_profile_avgt_list.append([audience_features_profile_avgt])
			audience_features_profile_veri_yes_list.append([audience_features_profile_veri_yes])
			audience_features_profile_veri_no_list.append([audience_features_profile_veri_no])

			author_at_name_dict = find_at_names_in_hist_tweets(hist_list)
			audience_at_name_dict = find_at_names_in_hist_tweets(audience_hist)

			message_sent = 0

			if author_full_name in audience_at_name_dict:
				message_sent = audience_at_name_dict[author_full_name]

			rank = 0.0

			if message_sent != 0:
				v = audience_at_name_dict.values()
				index = sorted(v).index(message_sent)
				rank = 	float(index + 1)/len(v) 

			mutual_meassage = 0

			if author_full_name in audience_at_name_dict:
				for one_aud in audience_full_name:
					if one_aud in author_at_name_dict:
						mutual_meassage = 1
						break

			audience_features_X.append([message_sent, rank, mutual_meassage])


			len_of_hist_list = 0
			rand_order = random.sample(range(1, len(hist_list)), len(hist_list)-1)
			modified_hist_list = ''

			while(len_of_hist_list<1000):
				try:
					tweet_add = hist_list[rand_order.pop()].encode('utf-8')
					len_of_hist_list += len(tweet_add)
					modified_hist_list = modified_hist_list +  tweet_add + '\n' 
				except IndexError:
					break
			
			audience_features_author_hist_list.append(modified_hist_list)

			len_of_hist_list = 0
			rand_order = random.sample(range(1, len(audience_hist)), len(audience_hist)-1)
			modified_hist_list = ''

			while(len_of_hist_list<1000):
				try:
					tweet_add = audience_hist[rand_order.pop()].encode('utf-8')
					len_of_hist_list += len(tweet_add)
					modified_hist_list = modified_hist_list +  tweet_add + '\n' 
				except IndexError:
					break
			
			audience_features_audience_hist_list.append(modified_hist_list)

			top_tfidf_list = []
			vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,1), stop_words='english')
			tfidf_matrix = vectorizer.fit_transform(audience_hist)
			idf = vectorizer.idf_
			scores =  dict(zip(vectorizer.get_feature_names(), idf))
			sortedList = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
			top_100_number = len(sortedList)
			if top_100_number < 100:
				for n in range(top_100_number):
					top_tfidf_list.append(sortedList[n][0])
			else:
				for n in range(100):
					top_tfidf_list.append(sortedList[n][0])
			
			for term in top_tfidf_list:
				if term not in audience_features_tfidf_dict:
					audience_features_tfidf_dict[term] = 1
				else:
					continue
			input_data = " ".join(top_tfidf_list)
			audience_features_fit_tfidf_list.append(input_data)


			# Tweet feature part
			Pronunciation_list.append([document["number_Polysyllables "],document["number_no_vowels"]])
			tweet_features_unigrams_list.append(word_unigrams)
			tweet_features_bigrams_list.append(word_unigrams)
			cluster1000_d = BrownCluster.get_cluster_dic()[1]
			cluster1000 = cluster1000_d.keys()
			bcword_unigrams = [0]*len(cluster1000)
			bcword_bigrams = [0]*len(bigram_dict_keys)
			for j in BC:
				bcword_unigrams[cluster1000.index(j)] = 1
			for j in range(len(BC)-1):
				bi_BC = (BC[j],BC[j+1])
				bcword_bigrams[bigram_dict_keys.index(bi_BC)] = 1

			bcunigrams_list.append(bcword_unigrams)
			bcbigrams_list.append(bcword_bigrams)
			two_words_arc_list = []
			two_BC_arc_list = []
			two_words_arc = document['two_words_arc']
			two_BC_arc = document['two_BC_arc']

			for one_arc in two_words_arc:
				if one_arc not in words_arc_dict:
					words_arc_dict[one_arc] = index
					index += 1
				two_words_arc_list.append(words_arc_dict[one_arc])
			X_list_word_arc.append(two_words_arc_list)

			for one_arc in two_BC_arc:
				if one_arc not in BC_arc_dict:
					BC_arc_dict[one_arc] = BC_index
					BC_index += 1
				two_BC_arc_list.append(BC_arc_dict[one_arc])
			X_list_BC_arc.append(two_BC_arc_list)

			row_feature = []
			for a in TagList:			
				a_ratio = a+'_ratio'
				if a== '$':
					a = 'numeral_pos'
					a_ratio = a+'_ratio'
				a_value = document[a]
				a_ratio_value = document[a_ratio]
				row_feature.append(a_value)
				row_feature.append(a_ratio_value)
			row_feature.append(document['lexical_density'])
			Part_of_speech_list.append(row_feature)

			cap_row_feature = []
			for a in TagList:			
				a_cap_count = a+'_cap_count'
				if a== '$':
					a_cap_count = 'numeral_pos_cap'
				a_value = document[a_cap_count]
				cap_row_feature.append(a_value)

			cap_row_feature.append(document['ini_cap_number'])
			cap_row_feature.append(document['all_cap_number'])

			Cap_list.append(cap_row_feature)
			this_sentence_sentiment = []
			this_sentence_sentiment.append(float(document["positivenode"]))
			this_sentence_sentiment.append(float(document["negativenode"]))
			whole_sentiment = document['tweet_whole_sentimentpredict']
			original_whole_sentiment[senti_dict[whole_sentiment]] = 1
			this_sentence_sentiment = this_sentence_sentiment + original_whole_sentiment
			Tweet_whole_sentiment_list.append(this_sentence_sentiment)
			Tweet_word_sentiment_list.append([document["effect_distance"],document["min_word_effect"],document["max_word_effect"],document["sentiment_distance"],document["min_word_senti"],document["max_word_senti"]])
			intensifier = document['intensifier']
			intensifier_list.append([intensifier])

			# end

			
			sarcasm_score = document['sarcasm_score']			
			target_list.append(int(sarcasm_score.encode('utf-8')))

			if author_full_name in name_dict:
				name_dict[author_full_name].append(len(target_list)-1)
			else:
				name_dict[author_full_name] = [len(target_list)-1]
		except ValueError:
			continue
	print '0'
	client.close()

	# tweet feature 
	unicount_vect = CountVectorizer(ngram_range=(1,1), lowercase = True,  stop_words='english', min_df=1)
	X_unigrams = unicount_vect.fit_transform(tweet_features_unigrams_list)
	bicount_vect = CountVectorizer(ngram_range=(2,2), lowercase = True, stop_words='english',  min_df=1)
	X_bigrams = bicount_vect.fit_transform(tweet_features_bigrams_list)

	X_bcunigrams = csr_matrix(np.array(bcunigrams_list))
	X_bcbigrams = csr_matrix(np.array(bcbigrams_list))
	X_author_features = hstack([X_unigrams,X_bigrams,X_bcunigrams,X_bcbigrams])
	del X_unigrams,X_bigrams,X_bcunigrams,X_bcbigrams
	X_author_features = csr_matrix(X_author_features)
	words_arc_feature_len = len(words_arc_dict.keys())
	print '2'
	for one_list in X_list_word_arc:
		zero_list = [0]*words_arc_feature_len
		for j in one_list:
			zero_list[j] = 1
		X_word.append(zero_list)

	BC_arc_feature_len = len(BC_arc_dict.keys())

	for one_list in X_list_BC_arc:
		zero_list = [0]*BC_arc_feature_len
		for j in one_list:
			zero_list[j] = 1
		X_BC.append(zero_list)

	X_word = csr_matrix(np.array(X_word))
	X_BC = csr_matrix(np.array(X_BC))
	X_author_features = hstack([X_author_features, X_BC, X_word])
	print '3'
	del X_word, X_BC
	X_POS = csr_matrix(np.array(Part_of_speech_list))
	X_Pron = csr_matrix(np.array(Pronunciation_list))
	X_Cap = csr_matrix(np.array(Cap_list))
	X_whole_senti = csr_matrix(np.array(Tweet_whole_sentiment_list))
	X_word_senti = csr_matrix(np.array(Tweet_word_sentiment_list))
	X_intensifier = csr_matrix(np.array(intensifier_list))
	X_author_features = hstack([X_author_features, X_POS, X_Pron, X_Cap, X_whole_senti, X_word_senti, X_intensifier])
	X_author_features = csr_matrix(X_author_features)
	del X_POS, X_Pron, X_Cap, X_whole_senti, X_word_senti, X_intensifier
	# end

	X_other = X_author_features
	del X_author_features
	X_other = csr_matrix(X_other)
	print '4'

	# audience features
	X_aud_hist_topics = BOW.Get_unigrams(audience_features_audience_hist_list)[1]
	X_all = [a+b for a,b in zip(audience_features_author_hist_list, audience_features_audience_hist_list)]
	unicount_vect = CountVectorizer(ngram_range=(1,1), lowercase = True, stop_words='english', min_df=1)
	unicount_vect.fit(X_all)
	audience_features_X_author = unicount_vect.transform(audience_features_author_hist_list)
	audience_features_X_audience = unicount_vect.transform(audience_features_audience_hist_list)

	vectorizer = CountVectorizer(min_df=1, stop_words='english')
	audience_features_X_unigrams = vectorizer.fit_transform(audience_features_unigrams_list)

	count_vect = CountVectorizer(vocabulary=audience_features_tfidf_dict.keys(),binary=True)
	audience_features_X_tfidf = count_vect.fit_transform(audience_features_fit_tfidf_list)


	print '5'
	audience_features_profile_nofer_list = preprocess_num(np.array(audience_features_profile_nofer_list))
	audience_features_profile_nofing_list = preprocess_num(np.array(audience_features_profile_nofing_list))
	audience_features_profile_not_list = preprocess_num(np.array(audience_features_profile_not_list))
	audience_features_profile_duration_list = preprocess_num(np.array(audience_features_profile_duration_list))
	audience_features_profile_avgt_list = preprocess_num(np.array(audience_features_profile_avgt_list))
	audience_features_profile_veri_yes_list = np.array(audience_features_profile_veri_yes_list)
	audience_features_profile_veri_no_list = np.array(audience_features_profile_veri_no_list)

	audience_features_X_profile_info = np.concatenate((audience_features_profile_nofer_list, audience_features_profile_nofing_list, audience_features_profile_duration_list,
						audience_features_profile_avgt_list, audience_features_profile_veri_yes_list, audience_features_profile_veri_no_list,
						audience_features_profile_not_list), axis=1)
	del audience_features_profile_nofer_list, audience_features_profile_nofing_list, audience_features_profile_duration_list,audience_features_profile_avgt_list, audience_features_profile_veri_yes_list, audience_features_profile_veri_no_list,audience_features_profile_not_list
	audience_features_X = np.hstack((np.array(audience_features_X),audience_features_X_profile_info))
	del audience_features_X_profile_info
	audience_features_X_his_com = csr_matrix(audience_features_X)
	audience_features_X_his_com = hstack([audience_features_X_his_com, audience_features_X_tfidf, audience_features_X_unigrams])
	del audience_features_X_tfidf, audience_features_X_unigrams
	audience_features_X_his_com = csr_matrix(audience_features_X_his_com)

	X_other = hstack([audience_features_X_his_com, X_other])
	del audience_features_X_his_com
	X_other = csr_matrix(X_other)
	print '6'
	Y = np.array(target_list)
	del target_list
	name_list = name_dict.keys()
	total_set_length = len(name_list)
	kf = KFold(total_set_length, n_folds=5)
	avg_score = []
	avg_auc_score = []
	for train_index, test_index in kf:
		temp_index = []
		for i in train_index:
			temp_index += name_dict[name_list[i]]
		train_index = np.array(temp_index)
		print '7'
		temp_index = []
		for j in test_index:
			temp_index += name_dict[name_list[j]]
		test_index = np.array(temp_index)
		print '8'
		
		X_aud_hist_topics_train, X_aud_hist_topics_test = X_aud_hist_topics[train_index], X_aud_hist_topics[test_index]
		model = lda.LDA(n_topics=100, n_iter=300, random_state=1)
		X_aud_hist_topics_train = csr_matrix(X_aud_hist_topics_train)
		model.fit(X_aud_hist_topics_train)
		X_aud_hist_topics_train = model.doc_topic_
		X_aud_hist_topics_train = csr_matrix(X_aud_hist_topics_train)
		X_aud_hist_topics_test = model.transform(X_aud_hist_topics_test)
		X_aud_hist_topics_test = csr_matrix(X_aud_hist_topics_test)

		X_train = X_aud_hist_topics_train
		X_test = X_aud_hist_topics_test
		del X_aud_hist_topics_test, X_aud_hist_topics_train
		######


		print '9'
		X_author_train, X_author_test = audience_features_X_author[train_index], audience_features_X_author[test_index]
		X_audience_train, X_audience_test = audience_features_X_audience[train_index], audience_features_X_audience[test_index]
		
		X_author_train_index = X_author_train.shape[0]
		audience_features_X_train = vstack([X_author_train, X_audience_train])
		audience_features_X_train = csr_matrix(audience_features_X_train)
		model = lda.LDA(n_topics=100, n_iter=300, random_state=1)
		model.fit(audience_features_X_train)
		del audience_features_X_train
		doc_topic = model.doc_topic_
		X_author_train = doc_topic[:X_author_train_index]
		X_audience_train = doc_topic[X_author_train_index:]
		del doc_topic
		print '10'
		audience_features_X_train = abs(X_author_train - X_audience_train)
		del X_author_train, X_audience_train
		min_max_scaler = preprocessing.MinMaxScaler()
		audience_features_X_train = min_max_scaler.fit_transform(audience_features_X_train)
		audience_features_X_train = csr_matrix(audience_features_X_train)
		X_train = hstack([X_train,audience_features_X_train])
		del audience_features_X_train

		print '11'
		X_author_test = X_author_test.toarray()
		X_audience_test = X_audience_test.toarray()
		X_author_test = model.transform(X_author_test)
		X_audience_test = model.transform(X_audience_test)
		audience_features_X_test = abs(X_author_test - X_audience_test)
		del X_author_test,X_audience_test
		min_max_scaler = preprocessing.MinMaxScaler()
		audience_features_X_test = min_max_scaler.fit_transform(audience_features_X_test)
		audience_features_X_test = csr_matrix(audience_features_X_test)
		X_test = hstack([X_test,audience_features_X_test])
		del audience_features_X_test
		print '12'
		#


		X_other_train, X_other_test = X_other[train_index], X_other[test_index]
		X_train = hstack([X_train, X_other_train])
		X_test = hstack([X_test, X_other_test])
		del X_other_train, X_other_test
		print '13'
		Y_train, Y_test = Y[train_index], Y[test_index]

		parameters = {'tol':[0.001,0.0001], 'C':[0.00001, 0.0001, 0.001, 0.1, 1, 10]}
		lr = linear_model.LogisticRegression(penalty='l2')
		clf = grid_search.GridSearchCV(lr, parameters)
		clf.fit(X_train, Y_train)
		y_true, y_pred = Y_test, clf.predict(X_test)
		score = clf.score(X_test, Y_test)
		avg_score.append(score)
		auc = roc_auc_score(y_true, y_pred)
		avg_auc_score.append(auc)
		print score, auc

	accuracy = reduce(lambda x, y: x + y, avg_score) / len(avg_score)
	auc_score = reduce(lambda x, y: x + y, avg_auc_score) / len(avg_auc_score)
	print 'accuracy is: %s, auc score is:%s'%(accuracy, auc_score)



if __name__ == '__main__':
	try:
		if sys.argv[1] == 'Author_historical_salient_terms':
			Author_historical_salient_terms()
		elif sys.argv[1] == 'Author_historical_sentiment':
			Author_historical_sentiment()
		elif sys.argv[1] == 'word_unigrams_bigrams':
			word_unigrams_bigrams()
		elif sys.argv[1] == 'intensifier':
			intensifier()
		elif sys.argv[1] == 'Part_of_speech':
			Part_of_speech()
		elif sys.argv[1] == 'Capitalization':
			Capitalization()
		elif sys.argv[1] == 'Pronunciation':
			Pronunciation()
		elif sys.argv[1] == 'Tweet_whole_sentiment':
			Tweet_whole_sentiment()
		elif sys.argv[1] == 'Tweet_word_sentiment':
			Tweet_word_sentiment()
		elif sys.argv[1] == 'Brown_Cluster_unigrams_bigrams':
			Brown_Cluster_unigrams_bigrams()
		elif sys.argv[1] == 'profile_unigrams':
			profile_unigrams()
		elif sys.argv[1] == 'Author_historical_topics':
			Author_historical_topics()
		elif sys.argv[1] == 'profile_information':
			profile_information()
		elif sys.argv[1] == 'dependency_arcs':
			dependency_arcs()
		elif sys.argv[1] == 'author_historical_sentiment':
			author_historical_sentiment()	
		elif sys.argv[1] == 'Audience_historical_topics':
			Audience_historical_topics()
		elif sys.argv[1] == 'Audience_historical_salient_terms':
			Audience_historical_salient_terms()
		elif sys.argv[1] == 'Audience_profile_unigrams':
			Audience_profile_unigrams()
		elif sys.argv[1] == 'Audience_profile_information':
			Audience_profile_information()
		elif sys.argv[1] == 'Audience_author_historical_communication':
			Audience_author_historical_communication()
		elif sys.argv[1] == 'original_tweet_unigrams':
			original_tweet_unigrams()
		elif sys.argv[1] == 'Pairwise_Brown_Cluster_original':
			Pairwise_Brown_Cluster_original()
		elif sys.argv[1] == 'interactional_topics':
			interactional_topics()
		elif sys.argv[1] == 'environment_features':
			environment_features()
		elif sys.argv[1] == 'audience_features':
			audience_features()
		elif sys.argv[1] == 'author_features':
			author_features()
		elif sys.argv[1] == 'tweet_features':
			tweet_features()
		elif sys.argv[1] == 'baseline':
			baseline()
		elif sys.argv[1] == 'Tweet_Environemnt_features':
			Tweet_Environemnt_features()
		elif sys.argv[1] == 'Tweet_Audience_Features':
			Tweet_Audience_Features()
		
		elif sys.argv[1] == 'all':
			print "word_unigrams_bigrams:"
			word_unigrams_bigrams()
			print "intensifier:"
			intensifier()
			print "Part_of_speech:"
			Part_of_speech()
			print "Capitalization:"
			Capitalization()
			print "Pronunciation:"
			Pronunciation()
			print "Tweet_whole_sentiment:"
			Tweet_whole_sentiment()
			print "Tweet_word_sentiment:"
			Tweet_word_sentiment()
			print "Brown_Cluster_unigrams_bigrams:"
			Brown_Cluster_unigrams_bigrams()
			print "profile_unigrams:"
			profile_unigrams()
			print "Author_historical_topics:"
			Author_historical_topics()
		else:
			print 'other mode'
	except KeyError:
	#except IndexError:
		print 'try again and add mode arguments'