# Run by: python Model_historical_feature.py salient_terms
import sys
import operator
import numpy as np
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn import grid_search
from sklearn import cross_validation
from scipy.sparse import csr_matrix

# Generate the top 100 tfidf terms
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
	count_vect = CountVectorizer(vocabulary=tfidf_dict.keys())
	del tfidf_dict
	X = count_vect.fit_transform(fit_tfidf_list)
	del fit_tfidf_list
	#X = csr_matrix(X).toarray()
	Y_target = np.array(target_list)
	del target_list
	print X.shape, Y_target.shape
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
		Y_train, Y_test = Y_target[train_index], Y_target[test_index]
		X_train_part, X_dev, Y_train_part, Y_dev = cross_validation.train_test_split(
			X_train, Y_train, test_size = 0.11, random_state = 0)

		parameters = {'tol':[0.01,0.1,0.001], 'C':[1, 10, 100]}
		lr = linear_model.LogisticRegression(penalty='l2')
		clf = grid_search.GridSearchCV(lr, parameters)
		clf.fit(X_dev, Y_dev)
		best_params = clf.best_params_
		del lr,clf
		print best_params
		tuned_clf = linear_model.LogisticRegression(penalty='l2', C=best_params['C'], tol=best_params['tol'])
		tuned_clf.fit(X_train_part, Y_train_part)
		y_true, y_pred = Y_test, tuned_clf.predict(X_test)
		score = tuned_clf.score(X_test, Y_test)
		del tuned_clf
		print 'score is ',score
		avg_score.append(score)
		auc = roc_auc_score(y_true, y_pred)
		print 'auc is', auc 
		avg_auc_score.append(auc)

	accuracy = reduce(lambda x, y: x + y, avg_score) / len(avg_score)
	auc_score = reduce(lambda x, y: x + y, avg_auc_score) / len(avg_auc_score)
	print 'accuracy is: %s, auc score is:%s'%(accuracy, auc_score)

if __name__ == '__main__':
	try:
		if sys.argv[1] == 'Author_historical_salient_terms':
			Author_historical_salient_terms()

		else:
			print 'other mode'

	except IndexError:
		print 'try again and add mode arguments'