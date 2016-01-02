# Collect all profiles of users. Get the bag of words (unigrams) of each profile.
# update each doucment in MongoDB with field "profile_unigrams_names" and "profile_unigrams"
from AllTweets import collect_profiles
from sklearn.feature_extraction.text import CountVectorizer
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
dbtest = db.test

vectorizer = CountVectorizer(min_df=1)
corpus = collect_profiles()
X = vectorizer.fit_transform(corpus)
names = vectorizer.get_feature_names()

for i in range(0, len(X.toarray())):
	profileAuthor = dbtest.find()[i]['author']
	for j in range(0, len(X.toarray()[i])):
		if X.toarray()[i][j] > 1:
			X.toarray()[i][j] = 1

	unigrams_count = list(X.toarray()[i])
	print unigrams_count
	print i
	print profileAuthor
	result = dbtest.update_one({"author": profileAuthor},
		{
		    "$set": {
                "profile_unigrams_names": names,
                "profile_unigrams": unigrams_count
        	}
		}
	)


