# Run by: python Model_historical_feature.py Author_historical_salient_terms
import sys
import operator
import numpy as np
import pickle
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn import grid_search
from sklearn import cross_validation
from scipy.sparse import csr_matrix, vstack

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
			tweetList
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
	#for i in xrange(1000):
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
	print 'accuracy is: %s, auc score is:%s'%(result[0], result[1])

#SGD logloss
# Logistic Regression Model with 10 layer Cross Validation. 
# trainning set(8/10), parameter development set(1/10), test set(1/10).
def LR_model(name_dict, X, Y):
	print "X and Y's shape are",X.shape, Y.shape
	name_list = name_dict.keys()
	total_set_length = len(name_list)
	kf = KFold(total_set_length, n_folds=10)
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
		X_train_part, X_dev, Y_train_part, Y_dev = cross_validation.train_test_split(
			X_train, Y_train, test_size = 0.11, random_state = 0)

		parameters = {'tol':[0.001,0.0001], 'C':[0.00001, 0.0001, 0.001, 0.1, 1, 10]}
		lr = linear_model.LogisticRegression(penalty='l2')
		clf = grid_search.GridSearchCV(lr, parameters)
		clf.fit(X_dev, Y_dev)
		best_params = clf.best_params_
		del lr,clf
		#print best_params
		tuned_clf = linear_model.LogisticRegression(penalty='l2', C=best_params['C'], tol=best_params['tol'])
		tuned_clf.fit(X_train_part, Y_train_part)
		y_true, y_pred = Y_test, tuned_clf.predict(X_test)
		score = tuned_clf.score(X_test, Y_test)
		del tuned_clf
		#print 'score is ',score
		avg_score.append(score)
		auc = roc_auc_score(y_true, y_pred)
		#print 'auc is', auc 
		avg_auc_score.append(auc)

	accuracy = reduce(lambda x, y: x + y, avg_score) / len(avg_score)
	auc_score = reduce(lambda x, y: x + y, avg_auc_score) / len(avg_auc_score)
	#print 'accuracy is: %s, auc score is:%s'%(accuracy, auc_score)
	return (accuracy,auc_score)


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
		else:
			print 'other mode'

	except IndexError:
		print 'try again and add mode arguments'